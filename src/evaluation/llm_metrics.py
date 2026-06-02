"""Métricas de evaluación RAG basadas en LLM judge (estilo RAGAS)."""

from __future__ import annotations

import os
import re

from src.generation.prompts import FAITHFULNESS_EVAL_TEMPLATE, RELEVANCE_EVAL_TEMPLATE
from src.logger import get_logger

logger = get_logger(__name__)


CONTEXT_PRECISION_EVAL_TEMPLATE = """Evalúa si el siguiente FRAGMENTO es relevante para responder la PREGUNTA.

PREGUNTA: {question}

FRAGMENTO:
{chunk_text}

Responde SOLO con un número entre 0.0 y 1.0 donde:
- 1.0 = el fragmento contiene información directamente útil para responder la pregunta
- 0.5 = el fragmento tiene información parcialmente relevante
- 0.0 = el fragmento no es relevante para la pregunta

Número:"""

CONTEXT_RECALL_EVAL_TEMPLATE = """Evalúa si el CONTEXTO contiene la información necesaria para producir la RESPUESTA ESPERADA.

RESPUESTA ESPERADA: {expected_answer}

CONTEXTO:
{context}

Responde SOLO con un número entre 0.0 y 1.0 donde:
- 1.0 = el contexto contiene toda la información de la respuesta esperada
- 0.5 = el contexto contiene parte de la información
- 0.0 = el contexto no contiene la información necesaria

Número:"""


class LLMJudge:
    """Uses an LLM to evaluate RAG quality metrics (RAGAS-style)."""

    def __init__(self, mock_mode: bool | None = None):
        from src.config import get_section

        cfg = get_section("generation")
        self._modelo = cfg.get("modelo", "claude-sonnet-4-6")
        self._mock_mode = mock_mode if mock_mode is not None else cfg.get("mock_mode", False)

        if not self._mock_mode and not os.getenv("ANTHROPIC_API_KEY"):
            self._mock_mode = True

        self._client = None
        if not self._mock_mode:
            try:
                import anthropic

                self._client = anthropic.Anthropic()
                logger.info("llm_judge_inicializado", modelo=self._modelo)
            except Exception as e:
                logger.warning("llm_judge_fallback_mock", error=str(e))
                self._mock_mode = True

    def _call_llm(self, prompt: str) -> str:
        if self._mock_mode:
            return "0.75"

        try:
            response = self._client.messages.create(
                model=self._modelo,
                max_tokens=10,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.warning("llm_judge_error", error=str(e))
            return "0.5"

    def _parse_score(self, text: str) -> float:
        match = re.search(r"([01]\.?\d*)", text)
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
        return 0.5

    def evaluar_fidelidad(self, respuesta: str, contexto: list[dict]) -> float:
        """LLM-based faithfulness: is the answer grounded in context?"""
        if not respuesta or not contexto:
            return 0.0
        context_text = "\n\n".join(r.get("texto", "")[:500] for r in contexto[:5])
        prompt = FAITHFULNESS_EVAL_TEMPLATE.substitute(
            context=context_text, answer=respuesta[:1000]
        )
        result = self._call_llm(prompt)
        return self._parse_score(result)

    def evaluar_relevancia(self, pregunta: str, respuesta: str) -> float:
        """LLM-based relevance: does the answer address the question?"""
        if not pregunta or not respuesta:
            return 0.0
        prompt = RELEVANCE_EVAL_TEMPLATE.substitute(
            question=pregunta, answer=respuesta[:1000]
        )
        result = self._call_llm(prompt)
        return self._parse_score(result)

    def evaluar_precision_contexto(self, pregunta: str, contexto: list[dict]) -> float:
        """LLM-based context precision: are the retrieved chunks relevant?"""
        if not pregunta or not contexto:
            return 0.0
        scores = []
        for chunk in contexto[:5]:
            prompt = CONTEXT_PRECISION_EVAL_TEMPLATE.format(
                question=pregunta,
                chunk_text=chunk.get("texto", "")[:500],
            )
            result = self._call_llm(prompt)
            scores.append(self._parse_score(result))
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    def evaluar_recall_contexto(
        self, respuesta_esperada: str, contexto: list[dict]
    ) -> float:
        """LLM-based context recall: does context contain needed info?"""
        if not respuesta_esperada or not contexto:
            return 0.0
        context_text = "\n\n".join(r.get("texto", "")[:500] for r in contexto[:5])
        prompt = CONTEXT_RECALL_EVAL_TEMPLATE.format(
            expected_answer=respuesta_esperada[:500],
            context=context_text,
        )
        result = self._call_llm(prompt)
        return self._parse_score(result)
