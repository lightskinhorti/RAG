"""Carga centralizada de configuración desde YAML y variables de entorno."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _ROOT / "configs" / "default.yaml"

_config_cache: dict[str, Any] | None = None


def load_config(path: Path | None = None) -> dict[str, Any]:
    global _config_cache
    if _config_cache is not None and path is None:
        return _config_cache

    config_path = path or _DEFAULT_CONFIG
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["generation"]["mock_mode"] = (
        os.getenv("MOCK_LLM", str(cfg["generation"].get("mock_mode", False))).lower()
        == "true"
    )
    cfg["vector_store"]["directorio_persistencia"] = os.getenv(
        "CHROMA_PERSIST_DIR",
        cfg["vector_store"]["directorio_persistencia"],
    )
    cfg["vector_store"]["nombre_coleccion"] = os.getenv(
        "CHROMA_COLLECTION_NAME",
        cfg["vector_store"]["nombre_coleccion"],
    )

    if path is None:
        _config_cache = cfg
    return cfg


def get_section(section: str) -> dict[str, Any]:
    return load_config()[section]
