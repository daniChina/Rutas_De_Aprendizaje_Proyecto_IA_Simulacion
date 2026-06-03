"""
llm_assisted.py — Fase 3
========================
Solver híbrido: usa puntuaciones de utilidad del LLM como función objetivo
del algoritmo clásico de optimización.

Flujo completo del sistema híbrido:
    cursos.json
        ↓  (Fase 1) load_instance()
    LearningPathProblem
        ↓  (Fase 2) LLMClient.score_course()  →  u(v) ∈ [1, 10]
    LearningPathProblem con utilidades
        ↓  (Fase 3) llm_assisted_solver()
    Ruta óptima S* ⊆ V

Estado: STUB — implementación completa en la Fase 3.
"""

from __future__ import annotations

from typing import List, Tuple

# TODO (Fase 3): importar dependencias reales
# from ..problem import LearningPathProblem
# from ..llm.client import LLMClient
# from .baseline import dp_knapsack_dag


def llm_assisted_solver(
    # problem: LearningPathProblem,
    # llm_client: LLMClient,
    # objetivo_usuario: str,
) -> Tuple[List[str], float]:
    """
    Solver híbrido LLM + DP:
      1. Evalúa semánticamente todos los cursos con el LLM → u(v).
      2. Ejecuta el solver DP con las utilidades asignadas.
      3. Devuelve la ruta óptima S* y su valor objetivo.

    TODO (Fase 3): implementar combinando LLMClient y dp_knapsack_dag.

    Returns:
        (lista_de_ids_seleccionados, valor_objetivo_total)
    """
    # PLACEHOLDER
    raise NotImplementedError("Solver híbrido — implementar en Fase 3.")
