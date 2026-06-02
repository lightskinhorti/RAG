"""Framework de evaluación RAG con métricas estándar.

Implementa cuatro métricas clave para evaluar la calidad del pipeline RAG:
- Fidelidad (faithfulness): ¿la respuesta está fundamentada en el contexto?
- Relevancia de respuesta (answer relevance): ¿responde la pregunta?
- Precisión del contexto (context precision): ¿los fragmentos son relevantes?
- Recall del contexto (context recall): ¿se recuperó toda la información necesaria?
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.config import get_section
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ResultadoEvaluacion:
    """Resultado de evaluar un par pregunta/respuesta."""

    id: str
    pregunta: str
    respuesta_generada: str
    respuesta_esperada: str
    fidelidad: float
    relevancia_respuesta: float
    precision_contexto: float
    recall_contexto: float
    score_global: float = field(init=False)

    def __post_init__(self) -> None:
        self.score_global = round(
            (
                self.fidelidad
                + self.relevancia_respuesta
                + self.precision_contexto
                + self.recall_contexto
            )
            / 4,
            4,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pregunta": self.pregunta,
            "respuesta_generada": self.respuesta_generada[:300],
            "respuesta_esperada": self.respuesta_esperada[:300],
            "metricas": {
                "fidelidad": self.fidelidad,
                "relevancia_respuesta": self.relevancia_respuesta,
                "precision_contexto": self.precision_contexto,
                "recall_contexto": self.recall_contexto,
                "score_global": self.score_global,
            },
        }


@dataclass
class InformeEvaluacion:
    """Informe agregado de la evaluación del sistema RAG."""

    resultados: list[ResultadoEvaluacion]
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    @property
    def fidelidad_media(self) -> float:
        return _media([r.fidelidad for r in self.resultados])

    @property
    def relevancia_media(self) -> float:
        return _media([r.relevancia_respuesta for r in self.resultados])

    @property
    def precision_media(self) -> float:
        return _media([r.precision_contexto for r in self.resultados])

    @property
    def recall_media(self) -> float:
        return _media([r.recall_contexto for r in self.resultados])

    @property
    def score_global_medio(self) -> float:
        return _media([r.score_global for r in self.resultados])

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "num_preguntas": len(self.resultados),
            "metricas_globales": {
                "fidelidad_media": round(self.fidelidad_media, 4),
                "relevancia_respuesta_media": round(self.relevancia_media, 4),
                "precision_contexto_media": round(self.precision_media, 4),
                "recall_contexto_media": round(self.recall_media, 4),
                "score_global_medio": round(self.score_global_medio, 4),
            },
            "resultados_detalle": [r.to_dict() for r in self.resultados],
        }

    def guardar(self, path: Path) -> None:
        """Persiste el informe en formato JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("informe_guardado", path=str(path))


# ---------------------------------------------------------------------------
# Evaluadores individuales
# ---------------------------------------------------------------------------


def evaluar_fidelidad(respuesta: str, contexto: list[dict]) -> float:
    """Mide si la respuesta está fundamentada en el contexto recuperado.

    Estrategia: solapamiento léxico entre términos clave de la respuesta
    y el texto del contexto. Score alto si los términos de la respuesta
    aparecen en el contexto.

    Args:
        respuesta: Texto generado por el LLM.
        contexto: Lista de fragmentos recuperados.

    Returns:
        Score entre 0.0 y 1.0.
    """
    if not respuesta or not contexto:
        return 0.0

    texto_contexto = " ".join(r.get("texto", "") for r in contexto).lower()
    palabras_respuesta = _extraer_palabras_clave(respuesta)

    if not palabras_respuesta:
        return 0.0

    hits = sum(1 for w in palabras_respuesta if w in texto_contexto)
    return round(hits / len(palabras_respuesta), 4)


def evaluar_relevancia_respuesta(pregunta: str, respuesta: str) -> float:
    """Mide si la respuesta es relevante para la pregunta planteada.

    Estrategia: solapamiento de términos clave de la pregunta en la respuesta.

    Args:
        pregunta: Pregunta original del usuario.
        respuesta: Texto generado por el LLM.

    Returns:
        Score entre 0.0 y 1.0.
    """
    if not pregunta or not respuesta:
        return 0.0

    palabras_pregunta = _extraer_palabras_clave(pregunta)
    if not palabras_pregunta:
        return 0.0

    respuesta_lower = respuesta.lower()
    hits = sum(1 for w in palabras_pregunta if w in respuesta_lower)
    return round(hits / len(palabras_pregunta), 4)


def evaluar_precision_contexto(pregunta: str, contexto: list[dict]) -> float:
    """Mide qué porcentaje de fragmentos recuperados son relevantes para la pregunta.

    Estrategia: un fragmento se considera relevante si contiene al menos
    un término clave de la pregunta.

    Args:
        pregunta: Pregunta original del usuario.
        contexto: Lista de fragmentos recuperados.

    Returns:
        Score entre 0.0 y 1.0.
    """
    if not pregunta or not contexto:
        return 0.0

    palabras_pregunta = _extraer_palabras_clave(pregunta)
    if not palabras_pregunta:
        return 0.0

    relevantes = 0
    for fragmento in contexto:
        texto = fragmento.get("texto", "").lower()
        if any(w in texto for w in palabras_pregunta):
            relevantes += 1

    return round(relevantes / len(contexto), 4)


def evaluar_recall_contexto(respuesta_esperada: str, contexto: list[dict]) -> float:
    """Mide si el contexto recuperado contiene la información necesaria para responder.

    Estrategia: solapamiento entre términos clave de la respuesta esperada
    y el texto del contexto recuperado.

    Args:
        respuesta_esperada: Respuesta de referencia del dataset de evaluación.
        contexto: Lista de fragmentos recuperados.

    Returns:
        Score entre 0.0 y 1.0.
    """
    if not respuesta_esperada or not contexto:
        return 0.0

    texto_contexto = " ".join(r.get("texto", "") for r in contexto).lower()
    palabras_esperadas = _extraer_palabras_clave(respuesta_esperada)

    if not palabras_esperadas:
        return 0.0

    hits = sum(1 for w in palabras_esperadas if w in texto_contexto)
    return round(hits / len(palabras_esperadas), 4)


# ---------------------------------------------------------------------------
# Evaluador orquestador
# ---------------------------------------------------------------------------


class EvaluadorRAG:
    """Orquesta la evaluación completa del sistema RAG."""

    def __init__(self, rag_pipeline=None):
        """
        Args:
            rag_pipeline: Función callable(pregunta) → (respuesta, contexto).
                          Si es None, se usan respuestas y contextos pasados directamente.
        """
        self._pipeline = rag_pipeline
        cfg = get_section("evaluation")
        self._dataset_path = Path(cfg.get("dataset_path", "data/evaluation/eval_dataset.json"))
        self._output_path = Path(cfg.get("output_path", "data/evaluation/results.json"))

    def evaluar_muestra(
        self,
        pregunta: str,
        respuesta_generada: str,
        respuesta_esperada: str,
        contexto: list[dict],
        sample_id: str = "q_manual",
    ) -> ResultadoEvaluacion:
        """Evalúa una muestra individual."""
        return ResultadoEvaluacion(
            id=sample_id,
            pregunta=pregunta,
            respuesta_generada=respuesta_generada,
            respuesta_esperada=respuesta_esperada,
            fidelidad=evaluar_fidelidad(respuesta_generada, contexto),
            relevancia_respuesta=evaluar_relevancia_respuesta(pregunta, respuesta_generada),
            precision_contexto=evaluar_precision_contexto(pregunta, contexto),
            recall_contexto=evaluar_recall_contexto(respuesta_esperada, contexto),
        )

    def evaluar_dataset(
        self,
        muestras: list[dict] | None = None,
    ) -> InformeEvaluacion:
        """Evalúa el dataset completo usando el pipeline RAG.

        Args:
            muestras: Lista de diccionarios con pregunta, respuesta_esperada y
                      opcionalmente respuesta_generada y contexto.
                      Si es None, carga desde el fichero de configuración.

        Returns:
            InformeEvaluacion con resultados agregados y por muestra.
        """
        if muestras is None:
            muestras = self._cargar_dataset()

        resultados = []
        for muestra in muestras:
            sample_id = muestra.get("id", f"q_{len(resultados)}")
            pregunta = muestra["pregunta"]
            respuesta_esperada = muestra.get("respuesta_esperada", "")

            respuesta_generada = muestra.get("respuesta_generada", "")
            contexto = muestra.get("contexto", [])

            if not respuesta_generada and self._pipeline:
                try:
                    resultado_pipeline = self._pipeline(pregunta)
                    respuesta_generada = resultado_pipeline.get("respuesta", "")
                    contexto = resultado_pipeline.get("contexto", [])
                except Exception as e:
                    logger.warning("error_pipeline_eval", id=sample_id, error=str(e))
                    respuesta_generada = ""

            resultado = self.evaluar_muestra(
                pregunta=pregunta,
                respuesta_generada=respuesta_generada,
                respuesta_esperada=respuesta_esperada,
                contexto=contexto,
                sample_id=sample_id,
            )
            resultados.append(resultado)
            logger.debug(
                "muestra_evaluada",
                id=sample_id,
                score_global=resultado.score_global,
            )

        informe = InformeEvaluacion(resultados=resultados)
        informe.guardar(self._output_path)

        logger.info(
            "evaluacion_completada",
            num_muestras=len(resultados),
            score_global=round(informe.score_global_medio, 4),
            fidelidad=round(informe.fidelidad_media, 4),
        )
        return informe

    def _cargar_dataset(self) -> list[dict]:
        """Carga el dataset de evaluación desde disco."""
        if not self._dataset_path.exists():
            logger.warning("dataset_no_encontrado", path=str(self._dataset_path))
            return []
        content = self._dataset_path.read_text(encoding="utf-8")
        return json.loads(content)


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------

_STOPWORDS_ES = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al",
    "en", "que", "es", "se", "no", "si", "con", "por", "para", "como", "una",
    "su", "sus", "pero", "más", "este", "esta", "estos", "estas", "o", "y", "a",
    "e", "ni", "lo", "le", "les", "me", "te", "nos", "os", "le", "cuando",
    "donde", "quien", "qué", "cómo", "cuándo", "cuál", "cuáles",
}


def _extraer_palabras_clave(texto: str, min_longitud: int = 4) -> list[str]:
    """Extrae palabras clave descartando stopwords y términos cortos."""
    palabras = re.findall(r"\b[a-záéíóúñü]+\b", texto.lower())
    return [w for w in palabras if len(w) >= min_longitud and w not in _STOPWORDS_ES]


def _media(valores: list[float]) -> float:
    """Calcula la media de una lista de valores."""
    return sum(valores) / len(valores) if valores else 0.0
