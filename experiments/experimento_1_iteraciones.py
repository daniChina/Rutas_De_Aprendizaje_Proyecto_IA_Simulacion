"""
experimento_1_iteraciones.py
============================
Experimento 1 — Impacto del número de iteraciones Monte Carlo

Hipótesis:
    A mayor número de iteraciones, la utilidad de la solución converge hacia
    un óptimo y el tiempo de ejecución crece de forma aproximadamente lineal.

Diseño:
    - Instancia fija: instancia_B_mediana (tamaño medio, buena señal/ruido).
    - Semilla fija (seed=42) para controlar la aleatoriedad entre configuraciones.
    - Se varía únicamente n_iter ∈ {50, 100, 250, 500, 1000, 2000, 5000}.
    - Por cada configuración se ejecutan N_REPS réplicas con semillas distintas
      para estimar media y desviación estándar de la utilidad.
    - Se registran: utilidad_media, utilidad_std, tiempo_medio_s, mejor_iter.

Salida:
    results/experimento_1_iteraciones.json
    results/experimento_1_iteraciones.csv
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any

# ── Añadir raíz del proyecto al path ─────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.instance import load_instance
from src.solver.mc_sampler import mc_path_sampler

logging.basicConfig(
    level=logging.WARNING,  # silenciar logs del solver durante el experimento
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("exp1")

# ── Parámetros del experimento ────────────────────────────────────────────────
INSTANCE_PATH = ROOT / "data/processed/instancia_B_evaluada.json"
ITERATIONS_SET = [50, 100, 250, 500, 1000, 2000, 5000]
TEMPERATURE    = 1.0      # temperatura fija para aislar la variable
N_REPS         = 10       # réplicas por configuración
BASE_SEED      = 42
RESULTS_DIR    = ROOT / "results"


def run() -> list[dict[str, Any]]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.warning("Cargando instancia: %s", INSTANCE_PATH)
    problem = load_instance(INSTANCE_PATH)

    # Verificar que todos los cursos tienen utilidad (precondición)
    if any(c.utilidad_relativa is None for c in problem.courses):
        raise RuntimeError(
            "La instancia no tiene utilidades asignadas. "
            "Ejecuta primero la evaluación LLM (Fase 2)."
        )

    print(f"\n{'='*60}")
    print(f"EXPERIMENTO 1 — Impacto de iteraciones Monte Carlo")
    print(f"Instancia : {problem.instance_id}")
    print(f"Cursos    : {len(problem.courses)} | T_max: {problem.t_max:.0f} h")
    print(f"Réplicas  : {N_REPS} por configuración")
    print(f"{'='*60}")
    print(f"{'n_iter':>8}  {'u_media':>8}  {'u_std':>7}  {'t_medio_s':>10}  {'mejor_iter_media':>17}")
    print(f"{'─'*8}  {'─'*8}  {'─'*7}  {'─'*10}  {'─'*17}")

    all_results: list[dict[str, Any]] = []

    for n_iter in ITERATIONS_SET:
        utilidades: list[float] = []
        tiempos:    list[float] = []
        conv_iters: list[int]   = []

        for rep in range(N_REPS):
            seed = BASE_SEED + rep * 100  # semillas distintas por réplica

            t_start = time.perf_counter()
            result = mc_path_sampler(
                problem,
                n_iterations=n_iter,
                temperature=TEMPERATURE,
                seed=seed,
            )
            elapsed = time.perf_counter() - t_start

            utilidades.append(result.objective_value)
            tiempos.append(elapsed)
            conv_iters.append(result.convergence_iter)

        u_media  = mean(utilidades)
        u_std    = stdev(utilidades) if len(utilidades) > 1 else 0.0
        t_medio  = mean(tiempos)
        ci_media = mean(conv_iters)

        print(
            f"{n_iter:>8}  {u_media:>8.2f}  {u_std:>7.3f}  "
            f"{t_medio:>10.4f}  {ci_media:>17.1f}"
        )

        all_results.append({
            "n_iter":              n_iter,
            "temperature":         TEMPERATURE,
            "n_reps":              N_REPS,
            "utilidad_media":      round(u_media, 4),
            "utilidad_std":        round(u_std, 4),
            "utilidad_min":        round(min(utilidades), 4),
            "utilidad_max":        round(max(utilidades), 4),
            "tiempo_medio_s":      round(t_medio, 6),
            "tiempo_total_s":      round(sum(tiempos), 6),
            "convergencia_media":  round(ci_media, 1),
            "instancia":           problem.instance_id,
            "n_cursos":            len(problem.courses),
            "t_max":               problem.t_max,
        })

    # ── Persistencia ──────────────────────────────────────────────────────────
    json_path = RESULTS_DIR / "experimento_1_iteraciones.json"
    csv_path  = RESULTS_DIR / "experimento_1_iteraciones.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"experimento": 1, "descripcion": "Impacto iteraciones MC", "resultados": all_results},
            f, ensure_ascii=False, indent=2,
        )

    # CSV plano para importar en hojas de cálculo / matplotlib
    headers = list(all_results[0].keys())
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in all_results:
            f.write(",".join(str(row[h]) for h in headers) + "\n")

    print(f"\n✓ Resultados guardados en:")
    print(f"    {json_path}")
    print(f"    {csv_path}")

    return all_results


if __name__ == "__main__":
    run()
