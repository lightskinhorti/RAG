"""Agente de investigación legal multi-paso con LangGraph."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AgentConfig:
    max_sub_queries: int = 3
    top_k_per_query: int = 5
    reranking: bool = True
    alpha: float = 0.7


class LegalResearchAgent:
    """Agente multi-paso que descompone preguntas legales complejas.

    Flujo:
    1. Analiza si la pregunta es simple o compleja.
    2. Si es compleja, descompone en sub-preguntas.
    3. Recupera documentos para cada sub-pregunta (o la pregunta directa).
    4. Sintetiza una respuesta unificada con citas.

    LangGraph se usa cuando está disponible; si no, cae a un pipeline
    secuencial equivalente.
    """

    def __init__(self, searcher, reranker, generador, config: AgentConfig | None = None):
        self._searcher = searcher
        self._reranker = reranker
        self._generador = generador
        self._config = config or AgentConfig()
        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        try:
            from typing import TypedDict

            from langgraph.graph import END, StateGraph

            class AgentState(TypedDict):
                pregunta_original: str
                sub_preguntas: list
                contexto_consolidado: list
                resultados_parciales: list
                respuesta_final: str
                pasos_ejecutados: list
                es_compleja: bool

            workflow = StateGraph(AgentState)
            workflow.add_node("analizar", self._analizar)
            workflow.add_node("descomponer", self._descomponer)
            workflow.add_node("recuperar_simple", self._recuperar_simple)
            workflow.add_node("recuperar_multiple", self._recuperar_multiple)
            workflow.add_node("sintetizar", self._sintetizar)

            workflow.set_entry_point("analizar")
            workflow.add_conditional_edges(
                "analizar",
                lambda s: "compleja" if s.get("es_compleja") else "simple",
                {"simple": "recuperar_simple", "compleja": "descomponer"},
            )
            workflow.add_edge("descomponer", "recuperar_multiple")
            workflow.add_edge("recuperar_simple", "sintetizar")
            workflow.add_edge("recuperar_multiple", "sintetizar")
            workflow.add_edge("sintetizar", END)

            logger.info("langgraph_graph_construido")
            return workflow.compile()

        except Exception as e:
            logger.warning("langgraph_no_disponible", error=str(e))
            return None

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------

    def _analizar(self, state: dict) -> dict:
        pregunta = state["pregunta_original"]
        signals = [
            " y " in pregunta.lower() and "?" in pregunta,
            pregunta.count("?") > 1,
            any(w in pregunta.lower() for w in ["compara", "diferencia", "relación entre", "versus"]),
            any(w in pregunta.lower() for w in ["además", "también", "aparte de", "junto con"]),
            len(pregunta.split()) > 25,
        ]
        es_compleja = sum(signals) >= 2
        logger.info("agente_analisis", es_compleja=es_compleja)
        return {
            **state,
            "es_compleja": es_compleja,
            "pasos_ejecutados": state.get("pasos_ejecutados", []) + ["analizar"],
        }

    def _descomponer(self, state: dict) -> dict:
        pregunta = state["pregunta_original"]
        sub_preguntas = self._descomponer_llm(pregunta) or self._descomponer_heuristica(pregunta)
        sub_preguntas = sub_preguntas[: self._config.max_sub_queries]
        logger.info("agente_descomposicion", num=len(sub_preguntas))
        return {
            **state,
            "sub_preguntas": sub_preguntas,
            "pasos_ejecutados": state.get("pasos_ejecutados", []) + ["descomponer"],
        }

    def _recuperar_simple(self, state: dict) -> dict:
        pregunta = state["pregunta_original"]
        resultados = self._buscar(pregunta)
        return {
            **state,
            "contexto_consolidado": resultados,
            "resultados_parciales": [{"pregunta": pregunta, "num_resultados": len(resultados)}],
            "pasos_ejecutados": state.get("pasos_ejecutados", []) + ["recuperar_simple"],
        }

    def _recuperar_multiple(self, state: dict) -> dict:
        sub_preguntas = state.get("sub_preguntas") or [state["pregunta_original"]]
        all_results: dict[str, dict] = {}
        parciales = []

        for sub_q in sub_preguntas:
            resultados = self._buscar(sub_q)
            parciales.append({"pregunta": sub_q, "num_resultados": len(resultados)})
            for r in resultados:
                doc_id = r.get("id", "")
                if doc_id not in all_results:
                    all_results[doc_id] = r
                else:
                    # boost docs that appear across multiple sub-queries
                    prev = all_results[doc_id].get("rrf_score", all_results[doc_id].get("score", 0))
                    new = r.get("rrf_score", r.get("score", 0))
                    all_results[doc_id]["rrf_score"] = prev + new * 0.5

        consolidated = sorted(
            all_results.values(),
            key=lambda x: x.get("rrf_score", x.get("score", 0)),
            reverse=True,
        )[: self._config.top_k_per_query * 2]

        logger.info("agente_recuperacion_multiple", docs=len(consolidated), sub_queries=len(sub_preguntas))
        return {
            **state,
            "contexto_consolidado": consolidated,
            "resultados_parciales": parciales,
            "pasos_ejecutados": state.get("pasos_ejecutados", []) + ["recuperar_multiple"],
        }

    def _sintetizar(self, state: dict) -> dict:
        gen = self._generador.generar(state["pregunta_original"], state.get("contexto_consolidado", []))
        return {
            **state,
            "respuesta_final": gen["respuesta"],
            "pasos_ejecutados": state.get("pasos_ejecutados", []) + ["sintetizar"],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _buscar(self, pregunta: str) -> list[dict]:
        self._searcher._alpha = self._config.alpha
        resultados = self._searcher.search(pregunta, top_k=self._config.top_k_per_query)
        if self._config.reranking and self._reranker and resultados:
            try:
                resultados = self._reranker.rerank(pregunta, resultados, self._config.top_k_per_query)
            except Exception:
                pass
        return resultados

    def _descomponer_llm(self, pregunta: str) -> list[str]:
        if self._generador._mock_mode or not getattr(self._generador, "_client", None):
            return []
        try:
            prompt = (
                f"Descompón la siguiente pregunta legal compleja en sub-preguntas simples "
                f"(máximo {self._config.max_sub_queries}, una por línea, numeradas).\n\n"
                f"Pregunta: {pregunta}\n\nSub-preguntas:"
            )
            response = self._generador._client.messages.create(
                model=self._generador._modelo,
                max_tokens=200,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            lines = response.content[0].text.strip().split("\n")
            return [
                re.sub(r"^\d+[\.\)]\s*", "", line.strip())
                for line in lines
                if line.strip() and len(line.strip()) > 10
            ][: self._config.max_sub_queries]
        except Exception as e:
            logger.warning("descomposicion_llm_error", error=str(e))
            return []

    def _descomponer_heuristica(self, pregunta: str) -> list[str]:
        parts = re.split(r"\s+(?:y|e|además|también)\s+", pregunta, flags=re.IGNORECASE)
        result = []
        for p in parts:
            p = p.strip().rstrip("?").strip() + "?"
            if len(p) > 15:
                result.append(p)
        return result or [pregunta]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, pregunta: str) -> dict:
        inicio = time.perf_counter()

        if self._graph is not None:
            initial = {
                "pregunta_original": pregunta,
                "sub_preguntas": [],
                "contexto_consolidado": [],
                "resultados_parciales": [],
                "respuesta_final": "",
                "pasos_ejecutados": [],
                "es_compleja": False,
            }
            try:
                final = self._graph.invoke(initial)
                latencia = (time.perf_counter() - inicio) * 1000
                logger.info("agente_completado", pasos=final.get("pasos_ejecutados"), latencia_ms=round(latencia, 1))
                return {
                    "pregunta": pregunta,
                    "respuesta": final.get("respuesta_final", ""),
                    "fuentes": final.get("contexto_consolidado", []),
                    "sub_preguntas": final.get("sub_preguntas", []),
                    "pasos": final.get("pasos_ejecutados", []),
                    "es_compleja": final.get("es_compleja", False),
                    "latencia_ms": round(latencia, 1),
                }
            except Exception as e:
                logger.error("agente_graph_error", error=str(e))

        # Fallback: simple linear pipeline
        resultados = self._buscar(pregunta)
        gen = self._generador.generar(pregunta, resultados)
        latencia = (time.perf_counter() - inicio) * 1000
        return {
            "pregunta": pregunta,
            "respuesta": gen["respuesta"],
            "fuentes": resultados,
            "sub_preguntas": [],
            "pasos": ["fallback_simple"],
            "es_compleja": False,
            "latencia_ms": round(latencia, 1),
        }
