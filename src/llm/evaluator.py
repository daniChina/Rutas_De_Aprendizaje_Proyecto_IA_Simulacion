"""
evaluator.py  (src/llm/evaluator.py)
=====================================
Orquestador de la evaluación semántica del catálogo completo de cursos.

Este módulo es el punto de integración entre la Fase 2 (LLM) y la Fase 3
(optimizador clásico). Recibe un LearningPathProblem, llama al LLMClient
para evaluar cada curso, actualiza los campos utilidad_relativa y justificacion
directamente en los objetos Course, y devuelve el problema enriquecido.

Flujo de datos:
    data/instances/instancia_X.json          (Fase 1)
        ↓ load_instance()
    LearningPathProblem (courses sin utilidad)
        ↓ evaluar_problema()                  (Fase 2 — este módulo)
    LearningPathProblem (courses con u(v) ∈ [1,10])
        ↓ guardar_problema_evaluado()
    data/processed/instancia_X_evaluada.json  →  Fase 3
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from ..problem import Course, LearningPathProblem
from .client import LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Función principal: evalúa todos los cursos de un problema
# ---------------------------------------------------------------------------

def evaluar_problema(
    problema: LearningPathProblem,
    objetivo_usuario: str,
    *,
    llm_client: LLMClient | None = None,
    delay_entre_llamadas: float = 0.5,
    puntuacion_fallback: int = 5,
) -> LearningPathProblem:
    """
    Evalúa semánticamente todos los cursos del problema con el LLM e incorpora
    las puntuaciones de utilidad u(v) ∈ [1,10] directamente en los objetos Course.

    Esta función es la interfaz principal de la Fase 2. El LearningPathProblem
    devuelto está listo para ser consumido por el algoritmo de optimización
    de la Fase 3 sin ninguna transformación adicional.

    Args:
        problema:             Instancia del problema con cursos sin evaluar.
        objetivo_usuario:     Meta de aprendizaje expresada en lenguaje natural.
        llm_client:           Cliente LLM. Si None, se instancia uno con la config
                              del entorno (útil para inyección de dependencias en tests).
        delay_entre_llamadas: Pausa en segundos entre llamadas para respetar rate limits.
        puntuacion_fallback:  Utilidad asignada a cursos que no se pudieron evaluar.

    Returns:
        El mismo LearningPathProblem con course.utilidad_relativa y
        course.justificacion actualizados para cada nodo.

    Raises:
        EnvironmentError: Si las credenciales no están configuradas y llm_client es None.
    """
    client = llm_client or LLMClient()

    total = len(problema.courses)
    exitosos = 0
    fallidos: list[str] = []

    logger.info("=" * 60)
    logger.info("Evaluacion del catalogo: instancia=%s", problema.instance_id)
    logger.info("Modelo LLM : %s", client.model)
    logger.info("Cursos     : %d", total)
    logger.info("T_max      : %.0f h", problema.t_max)
    logger.info("Objetivo   : %.100s%s", objetivo_usuario,
                "..." if len(objetivo_usuario) > 100 else "")
    logger.info("=" * 60)

    for i, course in enumerate(problema.courses, start=1):
        logger.info("Procesando [%d/%d] %s", i, total, course.id)

        curso_dict = {
            "id": course.id,
            "titulo": course.titulo,
            "descripcion": course.descripcion,
        }

        evaluacion = client.evaluar_curso(curso_dict, objetivo_usuario)

        if evaluacion is not None:
            # Actualizar el objeto Course directamente (mutación in-place)
            course.utilidad_relativa = evaluacion.utilidad_relativa
            course.justificacion = evaluacion.justificacion_breve
            exitosos += 1
        else:
            # Fallback: utilidad neutra para no bloquear el optimizador
            course.utilidad_relativa = puntuacion_fallback
            course.justificacion = (
                f"[FALLBACK] Evaluación no disponible. "
                f"Puntuación neutral {puntuacion_fallback}/10 asignada."
            )
            fallidos.append(course.id)
            logger.warning(
                "→ [%s] fallback u=%d/10", course.id, puntuacion_fallback
            )

        if i < total:
            time.sleep(delay_entre_llamadas)

    logger.info("=" * 60)
    logger.info("Completado: %d/%d exitosos | %d fallidos: %s",
                exitosos, total, len(fallidos), fallidos or "ninguno")
    logger.info("=" * 60)

    return problema


# ---------------------------------------------------------------------------
# Persistencia del problema evaluado
# ---------------------------------------------------------------------------

def guardar_problema_evaluado(
    problema: LearningPathProblem,
    directorio: str | Path = "data/processed",
) -> Path:
    """
    Guarda el LearningPathProblem enriquecido como JSON en data/processed/.

    El nombre del archivo sigue la convención:
        data/processed/<instance_id>_evaluada.json

    Este archivo es la entrada de la Fase 3 (algoritmo de optimización).

    Args:
        problema:    Instancia evaluada con utilidad_relativa en cada course.
        directorio:  Carpeta de destino (default: data/processed/).

    Returns:
        Path del archivo generado.
    """
    dir_path = Path(directorio)
    dir_path.mkdir(parents=True, exist_ok=True)

    output_path = dir_path / f"{problema.instance_id}_evaluada.json"

    datos = {
        "instance_id": problema.instance_id,
        "t_max": problema.t_max,
        "cursos": [
            {
                "id": c.id,
                "titulo": c.titulo,
                "descripcion": c.descripcion,
                "duracion_horas": c.duracion_horas,
                "prerrequisitos": c.prerrequisitos,
                "utilidad_relativa": c.utilidad_relativa,
                "justificacion_breve": c.justificacion,
                "evaluacion_exitosa": c.utilidad_relativa is not None
                                      and not (c.justificacion or "").startswith("[FALLBACK]"),
            }
            for c in problema.courses
        ],
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)

    logger.info(
        "Dataset evaluado guardado: %s (%d cursos)",
        output_path, len(problema.courses)
    )
    return output_path


# ---------------------------------------------------------------------------
# Vista previa de resultados (útil para el informe y la demo)
# ---------------------------------------------------------------------------

def resumen_evaluacion(
    problema: LearningPathProblem,
    top_n: int = 5,
) -> str:
    """
    Genera un resumen legible de la evaluación: top N cursos por utilidad.

    Args:
        problema: Problema con utilidades ya asignadas.
        top_n:    Número de cursos a mostrar en el ranking.

    Returns:
        String formateado listo para imprimir o incluir en el informe.
    """
    evaluados = [c for c in problema.courses if c.utilidad_relativa is not None]
    sin_evaluar = len(problema.courses) - len(evaluados)

    top = sorted(evaluados, key=lambda c: c.utilidad_relativa or 0, reverse=True)[:top_n]

    lineas = [
        f"\nEvaluación completada — Instancia: {problema.instance_id}",
        f"Cursos evaluados: {len(evaluados)} / {len(problema.courses)}"
        + (f"  ({sin_evaluar} sin evaluar)" if sin_evaluar else ""),
        f"\nTop {top_n} cursos más relevantes para el objetivo:",
        "─" * 60,
    ]

    for rango, curso in enumerate(top, start=1):
        lineas.append(
            f"{rango:2}. [{curso.id}] {curso.titulo}\n"
            f"    u={curso.utilidad_relativa}/10  |  {curso.duracion_horas} h\n"
            f"    {curso.justificacion}"
        )

    return "\n".join(lineas)
