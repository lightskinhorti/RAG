#!/usr/bin/env python3
"""Script CLI para descargar documentos reales del BOE e indexarlos.

Uso:
    python scripts/download_boe.py --dias 7
    python scripts/download_boe.py --fecha-inicio 2025-01-01 --fecha-fin 2025-01-31
    python scripts/download_boe.py --dias 30 --max-por-dia 20 --ingestar
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

# Asegura que el directorio raíz esté en el path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.boe_downloader import BOEDownloader
from src.logger import get_logger, setup_logging

setup_logging("info")
logger = get_logger("download_boe")


def main():
    parser = argparse.ArgumentParser(
        description="Descarga documentos reales del BOE para el sistema RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    grupo_fecha = parser.add_mutually_exclusive_group(required=True)
    grupo_fecha.add_argument(
        "--dias",
        type=int,
        metavar="N",
        help="Descargar los últimos N días del BOE",
    )
    grupo_fecha.add_argument(
        "--fecha-inicio",
        type=str,
        metavar="YYYY-MM-DD",
        help="Fecha de inicio (con --fecha-fin)",
    )

    parser.add_argument(
        "--fecha-fin",
        type=str,
        metavar="YYYY-MM-DD",
        default=None,
        help="Fecha de fin (por defecto: hoy)",
    )
    parser.add_argument(
        "--max-por-dia",
        type=int,
        default=15,
        metavar="N",
        help="Máximo de documentos por día (defecto: 15)",
    )
    parser.add_argument(
        "--directorio",
        type=str,
        default="data/raw",
        help="Directorio donde guardar los documentos (defecto: data/raw)",
    )
    parser.add_argument(
        "--ingestar",
        action="store_true",
        help="Indexar automáticamente los documentos descargados en ChromaDB",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Segundos de espera entre peticiones (defecto: 0.5)",
    )

    args = parser.parse_args()

    # Calcular rango de fechas
    hoy = date.today()
    if args.dias:
        fecha_inicio = hoy - timedelta(days=args.dias)
        fecha_fin = hoy
    else:
        try:
            fecha_inicio = date.fromisoformat(args.fecha_inicio)
            fecha_fin = date.fromisoformat(args.fecha_fin) if args.fecha_fin else hoy
        except ValueError as e:
            print(f"❌ Error en formato de fecha: {e}")
            sys.exit(1)

    if fecha_inicio > fecha_fin:
        print("❌ La fecha de inicio no puede ser posterior a la fecha de fin.")
        sys.exit(1)

    dias_total = (fecha_fin - fecha_inicio).days + 1
    print("\n📋 Configuración de descarga:")
    print(f"   Rango: {fecha_inicio} → {fecha_fin} ({dias_total} días)")
    print(f"   Máx. docs/día: {args.max_por_dia}")
    print(f"   Directorio: {args.directorio}")
    print(f"   Delay entre requests: {args.delay}s")
    print(f"   Indexar automáticamente: {'Sí' if args.ingestar else 'No'}")
    print()

    # Descarga
    downloader = BOEDownloader(
        raw_dir=args.directorio,
        max_docs_per_day=args.max_por_dia,
        delay_between_requests=args.delay,
    )

    print("🔄 Descargando documentos del BOE...")
    docs = downloader.descargar_rango(fecha_inicio, fecha_fin)

    print("\n✅ Descarga completada:")
    print(f"   Total documentos: {len(docs)}")
    if docs:
        print(f"   Longitud media: {sum(len(d.texto) for d in docs) // len(docs):,} caracteres")
        print(f"   Documentos en: {args.directorio}/")

    # Ingesta opcional
    if args.ingestar and docs:
        print(f"\n🔄 Indexando {len(docs)} documentos en ChromaDB...")
        try:
            from src.embeddings.embedder import get_embedder
            from src.ingestion.chunker import chunk_document
            from src.retrieval.vector_store import get_vector_store

            embedder = get_embedder()
            vector_store = get_vector_store()

            all_chunks = []
            for doc in docs:
                metadata = {
                    "fuente": doc.id,
                    "titulo": doc.titulo,
                    "fecha": doc.fecha,
                    "seccion": doc.seccion,
                    "departamento": doc.departamento,
                    "url": doc.url,
                }
                chunks = chunk_document(doc.texto, metadata=metadata)
                all_chunks.extend(chunks)

            print(f"   Total chunks generados: {len(all_chunks)}")
            embeddings = embedder.encode([c.texto for c in all_chunks])
            vector_store.add_chunks(all_chunks, embeddings)

            print(f"✅ Indexación completada. Total en vector store: {vector_store.count():,} chunks")
        except Exception as e:
            print(f"❌ Error durante la indexación: {e}")
            sys.exit(1)

    if not docs:
        print("⚠️  No se descargaron documentos. Verifica la conexión o las fechas.")
        sys.exit(1)

    print("\n🎉 ¡Listo! El sistema RAG ya tiene documentos reales del BOE.")
    if not args.ingestar:
        print("   Para indexarlos, ejecuta:")
        print(f"   python scripts/download_boe.py --dias {args.dias or dias_total} --ingestar")
        print("   O usa: make ingest")


if __name__ == "__main__":
    main()
