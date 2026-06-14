"""
cache.py — Fase 2
=================
Caché local de respuestas del LLM para evitar llamadas repetidas a la API.
Persiste las respuestas en un archivo JSON local (.llm_cache.json).

Estado: STUB — implementación completa en la Fase 2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class LLMCache:
    """
    Caché clave-valor que persiste respuestas del LLM en disco.
    La clave es el hash del prompt; el valor es la respuesta JSON del modelo.
    """

    def __init__(self, path: str = ".llm_cache.json") -> None:
        self._path = Path(path)
        self._data: dict = {}
        if self._path.exists():
            with self._path.open(encoding="utf-8") as f:
                self._data = json.load(f)

    def get(self, key: str) -> Optional[str]:
        """Devuelve la respuesta cacheada para una clave, o None si no existe."""
        return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        """Almacena una respuesta y persiste el caché en disco."""
        self._data[key] = value
        with self._path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def __len__(self) -> int:
        return len(self._data)

    def stats(self) -> dict:
        """Devuelve estadísticas básicas del caché para logging y el informe."""
        return {
            "path": str(self._path),
            "entries": len(self._data),
            "size_kb": round(self._path.stat().st_size / 1024, 1) if self._path.exists() else 0,
        }
