"""
experimento_1b_convergencia.py
===============================
Análisis complementario al Experimento 1 — Curva de convergencia MC

Mientras que exp1 mide el resultado FINAL con N iteraciones, este módulo
rastrea la utilidad ACUMULADA mejor en cada checkpoint a lo largo de una
sola ejecución larga, produciendo la curva de convergencia clásica que
muestra cuándo el algoritmo deja de mejorar.

Diseño:
  - Se ejecutan 3 réplicas de 5000 iteraciones sobre instancias A, B, C.
  - Se registra la mejor utilidad encontrada en cada checkpoint.
  - Se calcula la iteración de convergencia efectiva (último punto de mejora).
  - Salida: curvas listas para graficar en el informe (JSON + CSV).

Salida:
    results/experimento_1b_convergencia.json
    results/experimento_1b_convergencia.csv
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.instance import load_instance
from src.solver.mc_sampler import convergence_analysis

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("exp1b")

INSTANCES = [
    ("A", ROOT / "data/processed/instancia_A_evaluada.json"),
    ("B", ROOT / "data/processed/instancia_B_evaluada.json"),
    ("C", ROOT / "data/processed/instancia_C_evaluada.json"),
]
N_ITER      = 5000
N_REPS      = 3
BASE_SEED   = 42
CHECKPOINTS = [10, 25, 50, 100, 200, 350, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000]
RESULTS_DIR = ROOT / "results"


def run() -> list[dict[str, Any]]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("EXPERIMENTO 1b — Curvas de convergencia Monte Carlo")
    print(f"N_iter={N_ITER} | Réplicas={N_REPS} | checkpoints={len(CHECKPOINTS)}")
    print(f"{'='*60}")

    all_rows:   list[dict[str, Any]] = []
    curvas:     list[dict[str, Any]] = []

    for label, path in INSTANCES:
        if not path.exists():
            print(f"\n⚠  Instancia {label} no encontrada. Saltando.")
            continue

        problem = load_instance(path)

        if any(c.utilidad_relativa is None for c in problem.courses):
            print(f"\n⚠  Instancia {label} sin utilidades. Saltando.")
            continue

        print(f"\n── Instancia {label}: {problem.instance_id} "
              f"({len(problem.courses)} cursos) ──")

        # Acumular curvas de las N_REPS réplicas
        rep_curvas: list[dict[int, float]] = []

        for rep in range(1, N_REPS + 1):
            seed = BASE_SEED + rep * 17

            t0 = time.perf_counter()
            curva = convergence_analysis(
                problem,
                n_iterations=N_ITER,
                checkpoints=CHECKPOINTS,
                seed=seed,
            )
            elapsed = time.perf_counter() - t0

            rep_curvas.append(curva)

            # Registros individuales (para CSV)
            for ck, u in curva.items():
                all_rows.append({
                    "instancia_label": label,
                    "instancia_id":    problem.instance_id,
                    "n_cursos":        len(problem.courses),
                    "rep":             rep,
                    "seed":            seed,
                    "checkpoint":      ck,
                    "utilidad_mejor":  u,
                    "tiempo_total_s":  round(elapsed, 3),
                })

            print(f"  rep={rep} | t={elapsed:.2f}s | "
                  f"u@100={curva.get(100, '?')} | "
                  f"u@1000={curva.get(1000, '?')} | "
                  f"u@5000={curva.get(5000, '?')}")

        # ── Curva media entre réplicas ────────────────────────────────────────
        curva_media = {}
        curva_min   = {}
        curva_max   = {}
        for ck in CHECKPOINTS:
            vals = [rc.get(ck, 0) for rc in rep_curvas]
            curva_media[ck] = round(sum(vals) / len(vals), 3)
            curva_min[ck]   = min(vals)
            curva_max[ck]   = max(vals)

        # Iteración de convergencia efectiva: último checkpoint donde mejora > 1%
        u_final = curva_media[CHECKPOINTS[-1]]
        conv_ck = CHECKPOINTS[0]
        for ck in CHECKPOINTS[:-1]:
            u_here = curva_media[ck]
            u_next = curva_media.get(CHECKPOINTS[CHECKPOINTS.index(ck) + 1], u_here)
            if (u_next - u_here) / max(u_here, 1) > 0.01:  # mejora > 1%
                conv_ck = ck

        curvas.append({
            "instancia_label": label,
            "instancia_id":    problem.instance_id,
            "n_cursos":        len(problem.courses),
            "n_reps":          N_REPS,
            "n_iter":          N_ITER,
            "u_final_media":   u_final,
            "convergencia_ck": conv_ck,
            "curva_media":     {str(k): v for k, v in curva_media.items()},
            "curva_min":       {str(k): v for k, v in curva_min.items()},
            "curva_max":       {str(k): v for k, v in curva_max.items()},
        })

        print(f"  → Convergencia efectiva en checkpoint: {conv_ck} iter "
              f"| u_final={u_final}")

    # ── Persistencia ──────────────────────────────────────────────────────────
    json_path = RESULTS_DIR / "experimento_1b_convergencia.json"
    csv_path  = RESULTS_DIR / "experimento_1b_convergencia.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experimento":  "1b",
                "descripcion":  "Curvas de convergencia MC por instancia",
                "n_iter":       N_ITER,
                "n_reps":       N_REPS,
                "checkpoints":  CHECKPOINTS,
                "curvas":       curvas,
                "registros":    all_rows,
            },
            f, ensure_ascii=False, indent=2,
        )

    if all_rows:
        headers = list(all_rows[0].keys())
        with csv_path.open("w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
            for row in all_rows:
                f.write(",".join(str(row[h]) for h in headers) + "\n")

    print(f"\n✓ Resultados guardados en:")
    print(f"    {json_path}")
    print(f"    {csv_path}")

    return all_rows


if __name__ == "__main__":
    run()
