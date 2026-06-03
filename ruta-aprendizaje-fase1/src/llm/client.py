"""
client.py — Fase 2
==================
Wrapper de configuración y llamada al LLM para evaluación semántica de cursos.

Estado: STUB — implementación completa en la Fase 2.
Ver docs/fase2_llm_integracion.md para el diseño de prompts y arquitectura.
"""

from __future__ import annotations

import os
from typing import Optional

# TODO (Fase 2): importar openai, pydantic, tenacity
# from openai import OpenAI
# from .models import EvaluacionCurso
# from .prompts import construir_system_prompt, construir_user_prompt
# from .cache import LLMCache


class LLMClient:
    """
    Cliente LLM para evaluación semántica de cursos.

    Atributos configurables vía variables de entorno (.env):
        OPENAI_API_KEY  : Clave de la API.
        OPENAI_MODEL    : Modelo a usar (default: gpt-4o-mini).
        OPENAI_BASE_URL : URL base alternativa (Groq, Ollama, Azure).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        cache_path: Optional[str] = ".llm_cache.json",
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        # TODO (Fase 2): inicializar LLMCache y cliente OpenAI
        # self.cache = LLMCache(cache_path)
        # self._client = OpenAI(api_key=self.api_key)

    def score_course(self, course_id: str, descripcion: str, objetivo: str) -> float:
        """
        Evalúa semánticamente un curso y devuelve su utilidad u(v) ∈ [1, 10].

        TODO (Fase 2): implementar llamada real al LLM con:
          - construir_system_prompt() + construir_user_prompt()
          - response_format={"type": "json_object"}
          - Validación con EvaluacionCurso (Pydantic)
          - Reintentos con tenacity

        Args:
            course_id:   ID del curso (ej. "CS_601").
            descripcion: Texto del curso para análisis semántico.
            objetivo:    Meta de aprendizaje del usuario en lenguaje natural.

        Returns:
            Puntuación entera de utilidad [1, 10]. Placeholder: 5.
        """
        # PLACEHOLDER — retorna utilidad neutral hasta implementar la Fase 2
        return 5.0
