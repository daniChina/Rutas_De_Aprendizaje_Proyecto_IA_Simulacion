"""
instance.py
===========
Carga, filtrado y serialización de instancias del problema de rutas de aprendizaje.

Soporta dos modos de carga:
  1. Dataset completo (cursos.json) → LearningPathProblem con todos los cursos.
  2. Instancia de prueba (instancia_X.json) → subconjunto de cursos + T_max definido.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .problem import Course, LearningPathProblem


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_raw_dataset(path: str | Path) -> List[Course]:
    """
    Carga el dataset base (cursos.json) y devuelve la lista completa de cursos.

    Args:
        path: Ruta al archivo cursos.json.

    Returns:
        Lista de objetos Course con todos los campos del JSON.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        KeyError: Si algún curso no tiene los campos obligatorios (id, titulo, duracion_horas).
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw: List[Dict[str, Any]] = json.load(f)

    courses: List[Course] = []
    for item in raw:
        courses.append(
            Course(
                id=str(item["id"]),
                titulo=str(item["titulo"]),
                descripcion=str(item.get("descripcion", "")),
                duracion_horas=int(item["duracion_horas"]),
                prerrequisitos=list(item.get("prerrequisitos", [])),
                utilidad_relativa=item.get("utilidad_relativa"),
                justificacion=item.get("justificacion_breve"),
            )
        )
    return courses


def load_instance(path: str | Path) -> LearningPathProblem:
    """
    Carga una instancia de prueba desde su archivo JSON.

    El archivo de instancia puede tener dos formatos:
      A) Autónomo: incluye los datos completos de cada curso en "cursos".
      B) Por referencia: incluye solo "curso_ids" y una ruta al dataset base en "dataset_path".

    Args:
        path: Ruta al archivo de instancia (ej. data/instances/instancia_A_pequena.json).

    Returns:
        Objeto LearningPathProblem listo para el solver.

    Raises:
        FileNotFoundError: Si el archivo de instancia o el dataset base no existe.
        ValueError: Si el formato del archivo es inválido.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    instance_id = data.get("instance_id", path.stem)
    t_max = float(data["t_max"])

    # --- Formato A: cursos embebidos directamente ---
    if "cursos" in data:
        courses = [
            Course(
                id=str(c["id"]),
                titulo=str(c["titulo"]),
                descripcion=str(c.get("descripcion", "")),
                duracion_horas=int(c["duracion_horas"]),
                prerrequisitos=list(c.get("prerrequisitos", [])),
                utilidad_relativa=c.get("utilidad_relativa"),
                justificacion=c.get("justificacion_breve"),
            )
            for c in data["cursos"]
        ]

    # --- Formato B: referencia a dataset base + lista de IDs ---
    elif "curso_ids" in data:
        dataset_path = Path(data.get("dataset_path", "data/raw/cursos.json"))
        if not dataset_path.is_absolute():
            # Resolver relativo a la raíz del proyecto (dos niveles arriba de instances/)
            dataset_path = path.parent.parent.parent / dataset_path

        all_courses = {c.id: c for c in load_raw_dataset(dataset_path)}
        selected_ids: List[str] = data["curso_ids"]
        courses = [all_courses[cid] for cid in selected_ids if cid in all_courses]

    else:
        raise ValueError(
            f"Formato de instancia inválido en '{path}'. "
            "Debe contener 'cursos' (lista completa) o 'curso_ids' (lista de IDs)."
        )

    return LearningPathProblem(
        courses=courses,
        t_max=t_max,
        instance_id=instance_id,
    )


def load_full_problem(
    dataset_path: str | Path = "data/raw/cursos.json",
    t_max: float = 1295.0,
    instance_id: str = "full_dataset",
) -> LearningPathProblem:
    """
    Carga el dataset completo como un único problema (útil para depuración).

    Args:
        dataset_path: Ruta al dataset base.
        t_max: Restricción de tiempo (default: suma total de todos los cursos).
        instance_id: Identificador del problema.

    Returns:
        LearningPathProblem con los 35 cursos y el T_max indicado.
    """
    courses = load_raw_dataset(dataset_path)
    return LearningPathProblem(courses=courses, t_max=t_max, instance_id=instance_id)


# ---------------------------------------------------------------------------
# Serialización
# ---------------------------------------------------------------------------

def save_instance(problem: LearningPathProblem, path: str | Path) -> None:
    """
    Guarda una instancia del problema como archivo JSON autónomo.
    Útil para persistir instancias generadas programáticamente
    o para guardar el dataset enriquecido tras la evaluación del LLM (Fase 2).

    Args:
        problem: Instancia a serializar.
        path: Ruta de destino del archivo JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Any] = {
        "instance_id": problem.instance_id,
        "t_max": problem.t_max,
        "cursos": [
            {
                "id": c.id,
                "titulo": c.titulo,
                "descripcion": c.descripcion,
                "duracion_horas": c.duracion_horas,
                "prerrequisitos": c.prerrequisitos,
                "utilidad_relativa": c.utilidad_relativa,
                "justificacion_breve": c.justificacion,
            }
            for c in problem.courses
        ],
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
