"""
Submódulo LLM — Fase 2.

Exporta la interfaz pública del módulo para que otros módulos
puedan importar directamente desde src.llm sin conocer la estructura interna.
"""
from .client import LLMClient
from .evaluator import evaluar_problema, guardar_problema_evaluado, resumen_evaluacion
from .models import EvaluacionCurso

__all__ = [
    "LLMClient",
    "EvaluacionCurso",
    "evaluar_problema",
    "guardar_problema_evaluado",
    "resumen_evaluacion",
]
