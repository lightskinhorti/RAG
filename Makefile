.PHONY: help install download-data ingest run-api run-ui test lint docker-up docker-down clean

# ============================================================
# RAG Document Intelligence — Comandos de desarrollo
# ============================================================

PYTHON := python
PIP    := pip
DIAS   := 7

help: ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  Flujo rápido:"
	@echo "  1. make install"
	@echo "  2. make download-data"
	@echo "  3. make ingest"
	@echo "  4. make run-api  (en otra terminal: make run-ui)"

install: ## Instala todas las dependencias Python
	$(PIP) install -r requirements.txt

download-data: ## Descarga los últimos DIAS días del BOE (defecto: 7)
	$(PYTHON) scripts/download_boe.py --dias $(DIAS)

ingest: ## Indexa los documentos de data/raw en ChromaDB
	$(PYTHON) -c "
import sys; sys.path.insert(0, '.')
from pathlib import Path
from src.logger import setup_logging; setup_logging('info')
from src.ingestion.pipeline import ingest_directory
from src.embeddings.embedder import get_embedder
from src.retrieval.vector_store import get_vector_store

embedder = get_embedder()
vs = get_vector_store()
chunks = ingest_directory(Path('data/raw'))
if chunks:
    embeddings = embedder.encode([c.texto for c in chunks])
    vs.add_chunks(chunks, embeddings)
    print(f'✅ Indexados {len(chunks)} chunks en ChromaDB')
else:
    print('⚠️  No se encontraron documentos en data/raw')
    print('   Ejecuta: make download-data')
"

download-and-ingest: ## Descarga e indexa en un solo paso
	$(PYTHON) scripts/download_boe.py --dias $(DIAS) --ingestar

run-api: ## Inicia el servidor FastAPI (puerto 8000)
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-ui: ## Inicia la interfaz Streamlit (puerto 8501)
	streamlit run ui/app.py --server.port 8501

test: ## Ejecuta los tests unitarios
	pytest tests/ -v --tb=short

test-coverage: ## Tests con reporte de cobertura
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

lint: ## Comprueba el estilo del código
	@which ruff >/dev/null 2>&1 || pip install ruff -q
	ruff check src/ tests/ scripts/

eval: ## Ejecuta la evaluación del sistema RAG (requiere API activa)
	curl -s http://localhost:8000/evaluation | python -m json.tool

docker-up: ## Levanta todos los servicios con Docker Compose
	docker compose -f docker/docker-compose.yml up --build -d
	@echo "✅ Servicios disponibles:"
	@echo "   API:  http://localhost:8000/docs"
	@echo "   UI:   http://localhost:8501"
	@echo "   Logs: docker compose -f docker/docker-compose.yml logs -f"

docker-down: ## Detiene los servicios Docker
	docker compose -f docker/docker-compose.yml down

docker-logs: ## Muestra los logs de todos los servicios
	docker compose -f docker/docker-compose.yml logs -f

docker-rebuild: ## Reconstruye y reinicia los servicios
	docker compose -f docker/docker-compose.yml up --build --force-recreate -d

clean: ## Elimina archivos temporales y caché
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache htmlcov .coverage 2>/dev/null; true
	@echo "✅ Limpieza completada"

clean-data: ## ⚠️  Elimina la base de datos vectorial (requiere reindexar)
	@echo "⚠️  Esto eliminará data/chroma_db. ¿Continuar? [y/N]"
	@read ans && [ "$$ans" = "y" ] && rm -rf data/chroma_db && echo "✅ Eliminado" || echo "Cancelado"

stats: ## Muestra estadísticas de la colección (requiere API activa)
	@curl -s http://localhost:8000/stats | python -m json.tool

health: ## Comprueba el estado del servidor
	@curl -s http://localhost:8000/health | python -m json.tool
