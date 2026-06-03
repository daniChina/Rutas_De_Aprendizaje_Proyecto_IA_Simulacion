"""
prompts.py — Fase 2
===================
Ingeniería de prompts para la evaluación semántica de cursos.

Estado: STUB — implementación completa en la Fase 2.
Ver docs/fase2_llm_integracion.md para el diseño detallado con Few-Shot.
"""

from __future__ import annotations


def construir_system_prompt() -> str:
    """
    Devuelve el system prompt con rol, criterios de evaluación y ejemplos Few-Shot.

    TODO (Fase 2): implementar con:
      - Definición del rol experto
      - Tabla de criterios por rango de puntuación
      - 3 ejemplos Few-Shot calibrados (alta / media / baja relevancia)
      - Schema JSON de salida embebido
    """
    # PLACEHOLDER
    return (
        "Eres un experto en diseño curricular. "
        "Evalúa la relevancia del curso respecto al objetivo del usuario "
        "y devuelve un JSON con curso_id, utilidad_relativa (1-10) y justificacion_breve."
    )


def construir_user_prompt(objetivo_usuario: str, curso: dict) -> str:
    """
    Devuelve el user prompt para una llamada específica (objetivo + curso).

    TODO (Fase 2): implementar con formato estructurado y JSON embebido del curso.

    Args:
        objetivo_usuario: Meta de aprendizaje en lenguaje natural.
        curso: Diccionario con id, titulo y descripcion del curso.

    Returns:
        String del user prompt.
    """
    # PLACEHOLDER
    return (
        f"Objetivo: {objetivo_usuario}\n"
        f"Curso: {curso.get('titulo', '')} — {curso.get('descripcion', '')}"
    )
