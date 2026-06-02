# Arquitectura del Sistema RAG

## Visión General

El sistema implementa un pipeline RAG (Retrieval-Augmented Generation) completo sobre documentos legislativos españoles del BOE. La arquitectura sigue un diseño modular con separación clara entre ingesta, recuperación y generación.

## Diagrama de Flujo Principal

```
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE DE INGESTA                       │
│                                                             │
│  API BOE  →  boe_downloader  →  loader  →  chunker         │
│  (XML/PDF)    (requests)        (pypdf)    (3 estrategias)  │
│                                    ↓                         │
│                              embedder (sentence-transformers)│
│                                    ↓                         │
│                           ChromaDB (persist)                 │
└─────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE DE CONSULTA                      │
│                                                             │
│  Pregunta usuario                                           │
│       ↓                                                     │
│  embedder.encode(query)                                     │
│       ↓                    ↓                               │
│  Dense search           BM25 search                        │
│  (ChromaDB)             (rank-bm25)                        │
│       ↓                    ↓                               │
│      Reciprocal Rank Fusion (RRF)                          │
│                ↓                                            │
│      CrossEncoder Reranker (opcional)                       │
│                ↓                                            │
│      construir_prompt + contexto citado                     │
│                ↓                                            │
│      Claude (Anthropic API) / Mock                          │
│                ↓                                            │
│      Respuesta con citas [Fuente N]                        │
└─────────────────────────────────────────────────────────────┘
```

## Decisiones Técnicas

### 1. Modelo de Embeddings: `paraphrase-multilingual-MiniLM-L12-v2`

**Razón**: Los documentos del BOE están en español. Un modelo multilingüe captura mejor
la semántica del lenguaje jurídico español frente a modelos solo en inglés.
**Trade-off**: Algo más lento que `all-MiniLM-L6-v2` pero mucho mejor recall para consultas en español.

### 2. Búsqueda Híbrida: Embeddings + BM25

**Razón**: Los embeddings son buenos para similitud semántica pero malos para términos
exactos (e.g., "BOE-A-2024-1234", números de artículo). BM25 cubre este gap.
**Implementación**: Reciprocal Rank Fusion (RRF) en vez de interpolación lineal directa,
porque RRF es más robusto ante distribuciones de score diferentes entre los dos métodos.
**Parámetro `alpha`**: 0.7 por defecto (70% dense, 30% sparse). Ajustable por el usuario.

### 3. Reranking: CrossEncoder `ms-marco-MiniLM-L-6-v2`

**Razón**: Los bi-encoders (embeddings) son eficientes pero aproximados. Un cross-encoder
re-lee query+documento juntos y produce scores más precisos. Se aplica como paso final
sobre los top-K candidates.
**Trade-off**: +200-400ms de latencia, pero mejora precision@5 significativamente.

### 4. ChromaDB como Vector Store

**Razón**: Persistencia local sin dependencias externas. Para un portfolio/MVP es ideal.
**Extensibilidad**: La clase abstracta `VectorStore` permite swap a Pinecone o Weaviate
en producción sin cambiar el resto del código.

### 5. Tres Estrategias de Chunking

- **Fixed**: Baseline. Simple pero no respeta límites semánticos.
- **Recursive**: LangChain-style. Respeta párrafos > frases > palabras. Mejor para prosa legal.
- **Semantic**: Basado en oraciones. Mejor para documentos muy estructurados.

**Configurado como `recursive` por defecto** porque el lenguaje jurídico del BOE
tiene párrafos densos donde el salto de párrafo es el mejor separador natural.

### 6. Claude como LLM

**Razón**: Mejor comprensión del lenguaje jurídico en español que GPT-4 en benchmarks
informales. El modo mock permite desarrollo/demo sin consumir créditos.

## Estructura de Metadatos por Chunk

```python
{
    "fuente": "BOE-A-2025-1234",
    "titulo": "Real Decreto 123/2025, de 1 de enero...",
    "fecha": "2025-01-01",
    "seccion": "I. Disposiciones generales",
    "departamento": "Ministerio de Trabajo",
    "url": "https://www.boe.es/...",
    "chunk_index": 3,
    "estrategia": "recursive",
    "longitud_texto": 487
}
```

## Formato de Evaluación

El framework de evaluación implementa 4 métricas sin dependencia de LLM externo
(puramente léxicas), lo que garantiza reproducibilidad y velocidad:

| Métrica | Estrategia |
|---------|-----------|
| Fidelidad | Solapamiento léxico respuesta ↔ contexto |
| Relevancia | Solapamiento léxico pregunta ↔ respuesta |
| Precisión contexto | % fragmentos con términos de la pregunta |
| Recall contexto | Solapamiento respuesta esperada ↔ contexto |

Para producción se recomienda sustituir por métricas basadas en LLM judge (RAGAS).

## Seguridad y Consideraciones

- Las claves API nunca se hardcodean; siempre desde variables de entorno.
- El `.gitignore` excluye `data/raw/` y `data/chroma_db/` (pueden contener PII en corpus privados).
- La API incluye CORS configurado explícitamente para evitar orígenes no autorizados.
- Los errores se loguean con structlog en formato estructurado, sin exponer stacktraces al cliente.
