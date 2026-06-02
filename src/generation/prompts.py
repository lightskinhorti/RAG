"""Plantillas de prompt para el sistema RAG en español."""

from __future__ import annotations

from string import Template

# ---------------------------------------------------------------------------
# Sistema: define el rol y las restricciones del asistente
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Eres un asistente jurídico especializado en legislación española.
Tu función es responder preguntas basándote EXCLUSIVAMENTE en los fragmentos de documentos
legales del BOE (Boletín Oficial del Estado) que te proporcionan como contexto.

Reglas que debes seguir:
1. Responde únicamente con información presente en el contexto proporcionado.
2. Cita siempre la fuente usando el formato [Fuente N] al final de cada afirmación relevante.
3. Si la información no está en el contexto, indica claramente: "No encuentro información
   suficiente en los documentos disponibles para responder esta pregunta."
4. Sé preciso y conciso. Evita parafrasear en exceso el texto legal.
5. Cuando el texto legal sea ambiguo, indícalo explícitamente.
6. No inventes fechas, números de artículo, ni referencias legales."""

# ---------------------------------------------------------------------------
# Prompt principal de consulta
# ---------------------------------------------------------------------------

QUERY_TEMPLATE = Template("""A continuación tienes fragmentos de documentos legislativos
del BOE relevantes para responder la pregunta del usuario.

$context_block

---
Pregunta: $question

Responde de forma clara y precisa, citando las fuentes con [Fuente N] donde N es el
número del fragmento que respalda cada afirmación.""")

# ---------------------------------------------------------------------------
# Plantilla para el bloque de contexto
# ---------------------------------------------------------------------------

CONTEXT_ITEM_TEMPLATE = Template("""[Fuente $num] $titulo ($fecha)
Sección: $seccion | Departamento: $departamento
---
$texto""")

# ---------------------------------------------------------------------------
# Prompt para evaluación LLM (faithfulness)
# ---------------------------------------------------------------------------

FAITHFULNESS_EVAL_TEMPLATE = Template("""Evalúa si la RESPUESTA está fundamentada en
el CONTEXTO proporcionado.

CONTEXTO:
$context

RESPUESTA:
$answer

Responde SOLO con un número entre 0.0 y 1.0 donde:
- 1.0 = toda la respuesta está soportada por el contexto
- 0.5 = parte de la respuesta está soportada
- 0.0 = la respuesta no está soportada por el contexto

Número:""")

RELEVANCE_EVAL_TEMPLATE = Template("""Evalúa si la RESPUESTA es relevante para la PREGUNTA.

PREGUNTA: $question
RESPUESTA: $answer

Responde SOLO con un número entre 0.0 y 1.0 donde:
- 1.0 = la respuesta responde directamente la pregunta
- 0.5 = la respuesta es parcialmente relevante
- 0.0 = la respuesta no responde la pregunta

Número:""")


def construir_contexto(results: list[dict]) -> str:
    """Construye el bloque de contexto formateado para el prompt.

    Args:
        results: Lista de resultados de búsqueda con texto y metadata.

    Returns:
        Bloque de texto formateado con las fuentes numeradas.
    """
    bloques = []
    for i, result in enumerate(results, start=1):
        meta = result.get("metadata", {})
        bloque = CONTEXT_ITEM_TEMPLATE.substitute(
            num=i,
            titulo=meta.get("titulo", meta.get("fuente", "Documento desconocido"))[:120],
            fecha=meta.get("fecha", meta.get("fecha_publicacion", "fecha desconocida")),
            seccion=meta.get("seccion", "—"),
            departamento=meta.get("departamento", "—"),
            texto=result.get("texto", "")[:1200],
        )
        bloques.append(bloque)
    return "\n\n".join(bloques)


def construir_prompt_consulta(question: str, results: list[dict]) -> str:
    """Construye el prompt completo para una consulta RAG.

    Args:
        question: Pregunta del usuario.
        results: Lista de fragmentos recuperados con metadata.

    Returns:
        Prompt formateado listo para enviar al LLM.
    """
    context_block = construir_contexto(results)
    return QUERY_TEMPLATE.substitute(
        context_block=context_block,
        question=question,
    )
