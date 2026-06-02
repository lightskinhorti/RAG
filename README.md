# ⚖️ RAG Document Intelligence

> Sistema RAG de producción sobre legislación española del BOE: pregunta en lenguaje natural, respuesta citada con fuentes — evaluado con métricas estándar, desplegado con Docker Compose.

[![CI](https://github.com/lightskinhorti/RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/lightskinhorti/RAG/actions)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6F00)](https://trychroma.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**¿Qué problema resuelve?** La legislación española del BOE es extensa, técnica y difícil de consultar. Este sistema permite hacer preguntas en lenguaje natural y obtener respuestas precisas con citas a las fuentes originales, sobre documentos reales descargados vía API pública.

---

## 📊 Resultados del Benchmark

Evaluado sobre **50 preguntas** de legislación española (derecho laboral, protección de datos, tributario, administrativo y mercantil) con documentos reales del BOE.

| Métrica | Score | Descripción |
|---|---|---|
| **Fidelidad** | 0.82 | ¿La respuesta está fundamentada en el contexto recuperado? |
| **Relevancia** | 0.78 | ¿La respuesta responde directamente la pregunta? |
| **Precisión Contexto** | 0.85 | ¿Los fragmentos recuperados son relevantes? |
| **Recall Contexto** | 0.71 | ¿Se recuperó toda la información necesaria? |
| **Score Global** | **0.79** | Media de las 4 métricas |

| Métrica de Rendimiento | Valor |
|---|---|
| Latencia media (end-to-end) | ~800ms |
| Latencia p95 | ~1.4s |
| Corpus indexado | ~2.000+ chunks / ~150 documentos BOE |

> Reproducible: `make eval` genera `data/evaluation/results.json` con detalle por pregunta.

---

## ✨ Características Técnicas

| Componente | Implementación |
|---|---|
| **Embeddings** | `paraphrase-multilingual-MiniLM-L12-v2` (multilingüe, ideal para español) |
| **Vector Store** | ChromaDB con persistencia local + interfaz abstracta (swap a Pinecone) |
| **Búsqueda** | Híbrida: dense embeddings + BM25 sparse con Reciprocal Rank Fusion |
| **Reranking** | Cross-encoder `ms-marco-MiniLM-L-6-v2` para máxima precisión |
| **LLM** | Anthropic Claude con modo mock para demos sin API key |
| **Chunking** | 3 estrategias: fixed, recursive, semantic — configurables por YAML |
| **Filtrado** | Metadata filtering por sección, departamento y fecha del BOE |
| **Caché** | LRU cache en memoria para consultas repetidas |
| **Evaluación** | 4 métricas RAG: fidelidad, relevancia, precisión contexto, recall |
| **API** | FastAPI async con `asyncio.to_thread()`, streaming SSE, Pydantic v2 |
| **Observabilidad** | structlog con request_id, latencia por request, logs correlacionados |
| **UI** | Streamlit con diseño personalizado, chat con citas expandibles |
| **CI/CD** | GitHub Actions: tests + lint en cada push |
| **Deploy** | Docker Compose: un comando levanta API + UI + ChromaDB |

---

## 🚀 Inicio Rápido

### Opción A: Docker (recomendado)

```bash
git clone https://github.com/lightskinhorti/RAG.git && cd RAG
cp .env.example .env          # Añadir ANTHROPIC_API_KEY (opcional — funciona en mock)
docker compose -f docker/docker-compose.yml up --build -d
```

```bash
# Descargar e indexar 7 días del BOE
docker exec rag_api python scripts/download_boe.py --dias 7 --ingestar
```

Accede a: **UI →** http://localhost:8501 · **API Docs →** http://localhost:8000/docs

### Opción B: Desarrollo local

```bash
make install                   # Instalar dependencias
cp .env.example .env           # Configurar API key (opcional)
make download-and-ingest       # Descargar BOE + indexar
make run-api                   # API en http://localhost:8000
make run-ui                    # UI en http://localhost:8501 (otra terminal)
```

---

## 🏗️ Arquitectura

```mermaid
graph TD
    BOE["🌐 API BOE<br/>boe.es/datosabiertos"] --> DL[BOE Downloader]
    DL --> PARSE[XML Parser + Metadata]
    PARSE --> CHUNK[Chunker<br/>fixed / recursive / semantic]
    CHUNK --> EMB["Embeddings<br/>MiniLM-L12-v2 multilingual"]
    EMB --> DB[(ChromaDB)]

    Q["👤 Pregunta"] --> FILT{Metadata<br/>Filter?}
    FILT --> QEMB[Query Embedding]
    QEMB --> DENSE[Dense Search]
    QEMB --> BM25[BM25 Sparse]
    DENSE & BM25 --> RRF[Reciprocal Rank Fusion]
    RRF --> RANK[Cross-Encoder Reranker]
    RANK --> PROMPT["Prompt Builder<br/>con citas numeradas"]
    PROMPT --> LLM["Claude API / Mock"]
    DB --> DENSE
    LLM --> ANS["✅ Respuesta + Fuentes citadas"]

    CACHE["LRU Cache"] -.-> Q
    API["FastAPI async<br/>/query /ingest /stats"] --> Q
    UI["Streamlit UI<br/>Chat + Evaluación"] --> API
```

---

## 📁 Estructura del Proyecto

```
rag-document-intelligence/
├── src/
│   ├── ingestion/
│   │   ├── boe_downloader.py   # Descarga XML real del BOE sin autenticación
│   │   ├── loader.py           # Loaders para PDF, TXT, Markdown, XML
│   │   ├── chunker.py          # 3 estrategias de chunking
│   │   └── pipeline.py         # Orquestador de ingesta
│   ├── embeddings/
│   │   └── embedder.py         # sentence-transformers multilingüe
│   ├── retrieval/
│   │   ├── vector_store.py     # Abstracción ChromaDB + metadata filtering
│   │   ├── hybrid_search.py    # Dense + BM25 con RRF
│   │   └── reranker.py         # Cross-encoder reranking
│   ├── generation/
│   │   ├── prompts.py          # Templates de prompt en español
│   │   └── llm.py              # Anthropic Claude + modo mock
│   ├── evaluation/
│   │   └── metrics.py          # 4 métricas RAG + informe JSON
│   ├── api/
│   │   ├── main.py             # FastAPI: middleware, request_id, warm-up
│   │   ├── models.py           # Pydantic v2 models
│   │   └── routes/             # Endpoints async: /query /ingest /health /stats
│   ├── config.py               # Carga YAML + env vars
│   └── logger.py               # structlog con request_id
├── ui/
│   └── app.py                  # Streamlit con CSS personalizado
├── data/
│   ├── raw/                    # Documentos BOE descargados
│   ├── chroma_db/              # Vector store persistido
│   └── evaluation/
│       ├── eval_dataset.json   # 50 pares Q&A de referencia
│       └── results.json        # Resultados de evaluación
├── tests/
│   ├── test_ingestion.py       # Tests de carga y chunking
│   ├── test_embeddings.py      # Tests del módulo de embeddings
│   └── test_retrieval.py       # Tests de vector store, búsqueda y métricas
├── configs/
│   └── default.yaml            # Configuración global del sistema
├── docker/
│   ├── Dockerfile.api          # Multi-stage build para la API
│   ├── Dockerfile.ui           # Imagen para Streamlit
│   └── docker-compose.yml      # Orquestación completa
├── scripts/
│   └── download_boe.py         # CLI de descarga e ingesta
├── docs/
│   └── architecture.md         # Decisiones de diseño y trade-offs
├── .github/
│   └── workflows/ci.yml        # CI: tests + lint automáticos
├── Makefile                    # Comandos de desarrollo
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 🔌 API Reference

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/query` | Consulta RAG → respuesta + fuentes (con caché y filtros) |
| `POST` | `/query/stream` | Igual con streaming SSE |
| `POST` | `/ingest` | Indexar documentos desde directorio |
| `POST` | `/ingest/upload` | Subir y procesar un fichero |
| `GET` | `/health` | Estado del servicio |
| `GET` | `/stats` | Estadísticas de la colección |
| `GET` | `/evaluation` | Ejecutar evaluación completa |

**Ejemplo de consulta con filtrado por metadata:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "pregunta": "¿Cuál es la jornada laboral máxima en España?",
    "top_k": 5,
    "alpha": 0.7,
    "reranking": true,
    "filtro_seccion": "I. Disposiciones generales"
  }'
```

Todas las requests incluyen un header `X-Request-ID` para trazabilidad y `X-Process-Time-Ms` con la latencia.

---

## 📊 Evaluación del Sistema

El framework evalúa el pipeline completo con **50 preguntas** de referencia sobre legislación española, cubriendo 5 dominios jurídicos.

| Métrica | Descripción | Estrategia |
|---------|-------------|-----------|
| **Fidelidad** | ¿La respuesta está en el contexto? | Solapamiento semántico respuesta ↔ contexto |
| **Relevancia** | ¿Responde la pregunta? | Solapamiento semántico pregunta ↔ respuesta |
| **Precisión** | ¿Los chunks son relevantes? | % chunks con términos de la pregunta |
| **Recall** | ¿Se recuperó info suficiente? | Solapamiento respuesta esperada ↔ contexto |

```bash
make eval    # Ejecuta evaluación contra el dataset de 50 preguntas
```

---

## ⚙️ Configuración

Toda la configuración en `configs/default.yaml`. Variables de entorno sobrescriben YAML:

```yaml
embeddings:
  modelo: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

ingestion:
  estrategia_chunking: "recursive"  # fixed | recursive | semantic
  chunk_size: 512
  chunk_overlap: 64

retrieval:
  top_k: 5
  hybrid_alpha: 0.7               # 1.0=solo dense | 0.0=solo BM25
  reranking_habilitado: true

generation:
  modelo: "claude-sonnet-4-6"
  mock_mode: false                 # true = sin API key, respuestas simuladas
```

Variables de entorno (`.env`):
```env
ANTHROPIC_API_KEY=sk-ant-...   # Para respuestas reales del LLM
MOCK_LLM=false                 # true para demos sin API
```

---

## 🧪 Tests

```bash
make test              # Todos los tests
make test-coverage     # Con reporte de cobertura
make lint              # Comprobación de estilo con ruff
```

Los tests se ejecutan automáticamente en cada push vía [GitHub Actions](https://github.com/lightskinhorti/RAG/actions).

---

## 🛠️ Comandos Útiles

```bash
make help                 # Lista todos los comandos
make download-data        # Descarga últimos 7 días del BOE
make download-and-ingest  # Descarga + indexa en un paso
make run-api              # http://localhost:8000
make run-ui               # http://localhost:8501
make docker-up            # Todo con Docker
make eval                 # Ejecutar evaluación (API activa)
make stats                # Estadísticas de la colección
make health               # Health check del servidor
```

---

## 🔮 Mejoras Futuras

- [ ] Evaluación con RAGAS (LLM judge) para métricas de mayor fidelidad semántica
- [ ] Integración con Pinecone para escala cloud
- [ ] Fine-tuning del embedder sobre corpus jurídico español
- [ ] Caché semántico distribuido con Redis
- [ ] Monitorización con Prometheus + Grafana
- [ ] Agentes multi-step para consultas legales complejas (LangGraph)

---

## 👤 Autor

**Javier Hortigüela Valiente** — Data Engineer / ML Engineer

- GitHub: [@lightskinhorti](https://github.com/lightskinhorti)

---

## 📄 Licencia

MIT License. Ver [LICENSE](LICENSE).
