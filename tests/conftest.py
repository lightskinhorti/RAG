"""Configuración de pytest para el proyecto RAG."""

from __future__ import annotations

import sys
from pathlib import Path

# Asegura que el directorio raíz del proyecto esté en sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
