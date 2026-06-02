"""Integración con Anthropic Claude para generación de respuestas RAG.

Soporta modo real (API de Anthropic) y modo mock (sin llamadas externas),
con streaming opcional y extracción automática de citas.
"""

from __future__ import annotations

import os
import re
import time
from typing import Generator

from src.config import get_section
from src.generation.prompts import SYSTEM_PROMPT, construir_prompt_consulta
from src.logger import get_logger

logger = get_logger(__name__)


class GeneradorRespuestas:
    """Genera respuestas usando el LLM de Anthropic o un mock local."""

    def __init__(self, mock_mode: bool | None = None):
        cfg = get_section("generation")
        self._modelo = cfg.get("modelo", "claude-sonnet-4-6")
        self._max_tokens = cfg.get("max_tokens", 2048)
        self._temperatura = cfg.get("temperatura", 0.1)
        self._mock_mode = mock_mode if mock_mode is not None else cfg.get("mock_mode", False)

        if not self._mock_mode and not os.getenv("ANTHROPIC_API_KEY"):
            logger.warning(
                "anthropic_key_ausente",
                mensaje="ANTHROPIC_API_KEY no configurada. Activando modo mock.",
            )
            self._mock_mode = True

        self._client = None
        if not self._mock_mode:
            self._client = self._init_client()

    def _init_client(self):
        """Inicializa el cliente de Anthropic."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            logger.info("cliente_anthropic_inicializado", modelo=self._modelo)
            return client
        except ImportError as e:
            logger.error("anthropic_no_instalado", error=str(e))
            self._mock_mode = True
            return None

    def generar(
        self,
        pregunta: str,
        resultados: list[dict],
    ) -> dict:
        """Genera una respuesta completa con citas a partir de los fragmentos recuperados.

        Args:
            pregunta: Pregunta del usuario.
            resultados: Lista de fragmentos recuperados del vector store.

        Returns:
            Diccionario con respuesta, citas extraídas y metadatos.
        """
        inicio = time.time()
        prompt = construir_prompt_consulta(pregunta, resultados)

        if self._mock_mode:
            respuesta = self._generar_mock(pregunta, resultados)
        else:
            respuesta = self._llamar_api(prompt)

        latencia = (time.time() - inicio) * 1000
        citas = self._extraer_citas(respuesta)

        logger.info(
            "respuesta_generada",
            modo_mock=self._mock_mode,
            longitud_respuesta=len(respuesta),
            num_citas=len(citas),
            latencia_ms=round(latencia, 1),
        )

        return {
            "respuesta": respuesta,
            "citas": citas,
            "latencia_ms": round(latencia, 1),
            "modo_mock": self._mock_mode,
            "modelo": self._modelo if not self._mock_mode else "mock",
            "num_fuentes_usadas": len(resultados),
        }

    def generar_streaming(
        self,
        pregunta: str,
        resultados: list[dict],
    ) -> Generator[str, None, None]:
        """Genera una respuesta en modo streaming.

        Yields:
            Fragmentos de texto conforme los genera el LLM.
        """
        prompt = construir_prompt_consulta(pregunta, resultados)

        if self._mock_mode:
            yield from self._streaming_mock(pregunta, resultados)
            return

        try:
            with self._client.messages.stream(
                model=self._modelo,
                max_tokens=self._max_tokens,
                temperature=self._temperatura,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for texto in stream.text_stream:
                    yield texto
        except Exception as e:
            logger.error("error_streaming", error=str(e))
            yield f"\n[Error en streaming: {e}]"

    def _llamar_api(self, prompt: str) -> str:
        """Realiza la llamada a la API de Anthropic."""
        try:
            mensaje = self._client.messages.create(
                model=self._modelo,
                max_tokens=self._max_tokens,
                temperature=self._temperatura,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return mensaje.content[0].text
        except Exception as e:
            logger.error("error_api_anthropic", error=str(e), tipo=type(e).__name__)
            raise RuntimeError(f"Error llamando a la API de Anthropic: {e}") from e

    def _generar_mock(self, pregunta: str, resultados: list[dict]) -> str:
        """Genera una respuesta simulada para desarrollo y testing sin API."""
        if not resultados:
            return (
                "No encuentro información suficiente en los documentos disponibles "
                "para responder esta pregunta."
            )

        fragmentos = []
        for i, r in enumerate(resultados[:3], start=1):
            meta = r.get("metadata", {})
            titulo = meta.get("titulo", "Documento BOE")[:80]
            fragmentos.append(f"[Fuente {i}] {titulo}")

        intro = f"Basándome en los documentos legislativos disponibles sobre '{pregunta[:60]}':\n\n"
        cuerpo = "\n".join(
            f"• Según {f}, el documento contiene información relevante sobre este tema."
            for f in fragmentos
        )
        cierre = (
            "\n\n*Nota: Esta es una respuesta en modo mock. "
            "Configure ANTHROPIC_API_KEY para obtener respuestas reales del LLM.*"
        )
        return intro + cuerpo + cierre

    def _streaming_mock(
        self,
        pregunta: str,
        resultados: list[dict],
    ) -> Generator[str, None, None]:
        """Simula streaming con la respuesta mock."""
        respuesta = self._generar_mock(pregunta, resultados)
        palabras = respuesta.split(" ")
        for i, palabra in enumerate(palabras):
            yield palabra + (" " if i < len(palabras) - 1 else "")
            time.sleep(0.02)

    @staticmethod
    def _extraer_citas(respuesta: str) -> list[int]:
        """Extrae los números de fuente citados en la respuesta.

        Args:
            respuesta: Texto de la respuesta generada.

        Returns:
            Lista ordenada de índices de fuentes citadas.
        """
        numeros = re.findall(r"\[Fuente\s+(\d+)\]", respuesta, re.IGNORECASE)
        return sorted(set(int(n) for n in numeros))


def get_generador() -> GeneradorRespuestas:
    """Factoría para obtener el generador de respuestas configurado."""
    return GeneradorRespuestas()
