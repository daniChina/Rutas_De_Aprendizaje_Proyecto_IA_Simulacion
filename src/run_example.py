"""
run_example.py
==============
Script de demostración rápida para la Fase 1.

Ejecutar desde la raíz del proyecto:
    python src/run_example.py

Muestra:
  - Validación del DAG del dataset completo.
  - Resumen de cada instancia de prueba (A, B, C).
  - Orden topológico de cada instancia.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Añadir raíz del proyecto al path para imports relativos
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.instance import load_instance, load_full_problem

INSTANCES = [
    "data/instances/instancia_A_pequena.json",
    "data/instances/instancia_B_mediana.json",
    "data/instances/instancia_C_grande.json",
]


def separator(title: str = "") -> None:
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * pad}")
    else:
        print("─" * width)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Dataset completo
    # ------------------------------------------------------------------
    separator("DATASET COMPLETO")
    full = load_full_problem("data/raw/cursos.json", t_max=300.0, instance_id="completo")
    print(full.summary())

    topo = full.topological_order()
    print(f"\nOrden topológico ({len(topo)} nodos):")
    # Imprimir de a 7 por fila para mayor legibilidad
    for i in range(0, len(topo), 7):
        print("  " + "  →  ".join(topo[i : i + 7]))

    # ------------------------------------------------------------------
    # 2. Instancias de prueba
    # ------------------------------------------------------------------
    for instance_path in INSTANCES:
        path = Path(instance_path)
        if not path.exists():
            print(f"\n[AVISO] Instancia no encontrada: {instance_path}")
            continue

        separator(path.stem.upper())
        problem = load_instance(instance_path)
        print(problem.summary())

        # Verificar un ejemplo de selección ficticia (primeros N cursos según orden topológico)
        topo_local = problem.topological_order()
        # Tomar los primeros cursos del orden topológico hasta llenar aprox. 60% del T_max
        budget_target = problem.t_max * 0.6
        selection = []
        accumulated = 0.0
        for cid in topo_local:
            course = problem.get_course(cid)
            if course and accumulated + course.duration <= budget_target:
                selection.append(cid)
                accumulated += course.duration

        valid = problem.is_valid_selection(selection)
        obj = problem.objective_value(selection)
        dur = problem.selection_duration(selection)

        print(f"\nSelección de ejemplo ({len(selection)} cursos, {dur:.0f} h / {problem.t_max:.0f} h):")
        for cid in selection:
            c = problem.get_course(cid)
            print(f"  [{cid}] {c.titulo}  ({c.duracion_horas} h, u={c.utility:.1f})")
        print(f"Válida: {'✓' if valid else '✗'}  |  Utilidad total: {obj:.1f}")

    separator()
    print("\n✓ Fase 1 verificada. Todas las instancias cargadas y el DAG es válido.")
    print("  Próximo paso → Fase 2: ejecutar src/llm/client.py para poblar utilidades.\n")


if __name__ == "__main__":
    main()
