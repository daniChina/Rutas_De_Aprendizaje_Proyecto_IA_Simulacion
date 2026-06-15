#!/usr/bin/env python3
"""
run_fase3.py
============
Ejecuta el solver híbrido (LLM + DP/Greedy/MC) sobre las tres instancias reales
del proyecto: A, B y C.

Requiere que los archivos evaluados estén en:
    data/processed/instancia_A_evaluada.json
    data/processed/instancia_B_evaluada.json
    data/processed/instancia_C_evaluada.json

El script:
  1. Carga cada instancia.
  2. Para instancias A y B (pequeñas) permite DP exacto (bitmask).
  3. Para instancia C (35 cursos) fuerza Monte Carlo (force_mc_only=True)
     porque la DP exacta es inviable y la heurística no es fiable.
  4. Imprime en consola el resumen de la mejor ruta y guarda la solución
     en outputs/ruta_<instancia>.json.
  5. Además, guarda un resumen comparativo en outputs/comparativa_fase3.json.

Uso:
    python src/run_fase3.py
    python src/run_fase3.py --instancia A     # solo la instancia A
    python src/run_fase3.py --help
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Añadir el directorio raíz al path (por si ejecutamos desde src/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.problem import Course, LearningPathProblem
from src.solver.llm_assisted import llm_assisted_solver
from src.solver.baseline import DPResult

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_fase3")

# El mismo objetivo que se usó en la Fase 2 (debe ser coherente con las evaluaciones)
OBJETIVO_USUARIO = (
    "Quiero construir sistemas de inteligencia artificial aplicados al procesamiento "
    "de lenguaje natural, con enfoque en modelos grandes (LLMs) y despliegue en "
    "entornos de producción."
)


# -----------------------------------------------------------------------------
# Carga de instancias evaluadas
# -----------------------------------------------------------------------------

def cargar_instancia_evaluada(instancia_id: str) -> LearningPathProblem:
    """
    Carga un problema desde data/processed/<instancia_id>_evaluada.json
    Asume que el archivo tiene el formato generado por evaluator.guardar_problema_evaluado()
    """
    path = Path(f"data/processed/{instancia_id}_evaluada.json")
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Construir objetos Course
    cursos = []
    for c in data["cursos"]:
        # Los campos esperados: id, titulo, descripcion, duracion_horas, prerrequisitos,
        # utilidad_relativa, justificacion_breve (opcional)
        curso = Course(
            id=c["id"],
            titulo=c["titulo"],
            descripcion=c["descripcion"],
            duracion_horas=c["duracion_horas"],
            prerrequisitos=c["prerrequisitos"],
            utilidad_relativa=c.get("utilidad_relativa"),  # puede ser None si no está evaluada
        )
        cursos.append(curso)

    t_max = data.get("t_max", 300.0)  # por defecto 300h para instancia C
    instance_id = data.get("instance_id", instancia_id)

    problema = LearningPathProblem(cursos, t_max=t_max, instance_id=instance_id)

    # Verificar que todos los cursos tengan utilidad asignada (deberían, tras Fase 2)
    sin_utilidad = [c.id for c in problema.courses if c.utility is None]
    if sin_utilidad:
        logger.warning(
            "Los siguientes cursos no tienen utilidad asignada: %s. "
            "El solver no podrá optimizar correctamente.",
            sin_utilidad
        )
    return problema


# -----------------------------------------------------------------------------
# Guardar resultados
# -----------------------------------------------------------------------------

def guardar_ruta(instancia_id: str, resultado, problema: LearningPathProblem) -> None:
    """Guarda la mejor ruta en formato JSON dentro de outputs/."""
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    ruta_data = {
        "instancia": instancia_id,
        "solver_used": resultado.solver_used,
        "selected_courses": [],
        "total_utility": resultado.objective_value,
        "total_duration": resultado.total_duration,
        "t_max": problema.t_max,
        "utilization": resultado.total_duration / problema.t_max if problema.t_max > 0 else 0,
        "n_courses": len(resultado.selected_ids),
    }
    for cid in resultado.selected_ids:
        course = problema.get_course(cid)
        ruta_data["selected_courses"].append({
            "id": course.id,
            "titulo": course.titulo,
            "duracion_horas": course.duration,
            "utilidad_relativa": course.utility,
            "justificacion_breve": getattr(course, "justification", ""),
        })

    out_file = output_dir / f"ruta_{instancia_id}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(ruta_data, f, ensure_ascii=False, indent=2)
    logger.info("Ruta guardada en %s", out_file)


def guardar_comparativa(resultados: dict) -> None:
    """Guarda un resumen comparativo de las tres instancias."""
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    out_file = output_dir / "comparativa_fase3.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    logger.info("Comparativa guardada en %s", out_file)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ejecuta el solver híbrido sobre las instancias reales.")
    parser.add_argument(
        "--instancia",
        choices=["A", "B", "C", "all"],
        default="all",
        help="Instancia a ejecutar (por defecto todas)."
    )
    parser.add_argument(
        "--force_mc",
        action="store_true",
        help="Forzar Monte Carlo incluso para instancias pequeñas (útil para pruebas)."
    )
    args = parser.parse_args()

    instancias = []
    if args.instancia == "all":
        instancias = ["instancia_A", "instancia_B", "instancia_C"]
    else:
        instancias = [f"instancia_{args.instancia}"]

    resultados_globales = {}

    for inst_id in instancias:
        print("\n" + "=" * 70)
        print(f"  Resolviendo {inst_id.upper()}")
        print("=" * 70)

        try:
            problema = cargar_instancia_evaluada(inst_id)
        except FileNotFoundError as e:
            logger.error(e)
            continue

        # Decidir parámetros según instancia
        n = len(problema.courses)
        if args.force_mc:
            force_mc = True
        else:
            # Forzar MC solo en instancia C (35 cursos) porque DP exacto no es práctico
            # y la DP heurística puede dar resultados no confiables.
            force_mc = (inst_id == "instancia_C")
            if force_mc:
                logger.info("Instancia %s: forzando Monte Carlo por tamaño (%d cursos)", inst_id, n)
            else:
                logger.info("Instancia %s: usando solver híbrido (DP exacto para n<=20, sino heurístico)", inst_id)

        # Parámetros Monte Carlo: más iteraciones para instancia C
        mc_iter = 5000 if inst_id == "instancia_C" else 2000
        mc_temp = 1.0

        resultado = llm_assisted_solver(
            problem=problema,
            objetivo_usuario=OBJETIVO_USUARIO,
            llm_client=None,                     # No necesitamos evaluar otra vez
            mc_iterations=mc_iter,
            mc_temperature=mc_temp,
            granularidad_dp=1,
            force_mc_only=force_mc,
        )

        # Mostrar resumen
        print("\n" + resultado.summary())
        print("\n✅ Mejor ruta encontrada:")
        for idx, cid in enumerate(resultado.selected_ids, 1):
            curso = problema.get_course(cid)
            print(f"  {idx:2}. {cid} - {curso.titulo[:60]}")
            print(f"       utilidad={curso.utility}/10, duración={curso.duration:.0f}h")
            if hasattr(curso, "justification") and curso.justification:
                print(f"       justificación: {curso.justification[:100]}...")

        # Guardar ruta individual
        guardar_ruta(inst_id, resultado, problema)

        # Recolectar métricas para comparativa global
        resultados_globales[inst_id] = {
            "solver_used": resultado.solver_used,
            "total_utility": resultado.objective_value,
            "total_duration": resultado.total_duration,
            "n_courses": len(resultado.selected_ids),
            "t_max": problema.t_max,
            "utilization": resultado.total_duration / problema.t_max if problema.t_max > 0 else 0,
            "dp_utility": resultado.dp_result.objective_value if resultado.dp_result else None,
            "greedy_utility": resultado.greedy_result.objective_value,
            "mc_utility": resultado.mc_result.objective_value,
        }

    if resultados_globales:
        guardar_comparativa(resultados_globales)
        print("\n" + "=" * 70)
        print("  Resumen final (comparativa)")
        print("=" * 70)
        for inst, stats in resultados_globales.items():
            print(f"{inst.upper()}: u={stats['total_utility']:.1f} | {stats['n_courses']} cursos | "
                  f"{stats['total_duration']:.0f}h / {stats['t_max']:.0f}h | solver={stats['solver_used']}")
    else:
        logger.error("No se procesó ninguna instancia.")


if __name__ == "__main__":
    main()