"""
run_experiments.py
==================
Runner principal — ejecuta los 4 experimentos (+ análisis de convergencia 1b)
en secuencia y produce un resumen consolidado en JSON y un informe de texto.

Uso:
    python experiments/run_experiments.py               # todos
    python experiments/run_experiments.py --exp 1 3     # solo 1 y 3
    python experiments/run_experiments.py --list        # muestra disponibles

Precondiciones:
    - Las instancias A, B, C deben existir en data/instances/.
      Si no existen, ejecuta primero:
          python experiments/generate_instances.py
    - Las instancias deben tener utilidades asignadas (Fase 2 ejecutada).
      Si no las tienen, ejecuta:
          python src/run_fase2.py   (para cada instancia)
    - Si alguna instancia falta o carece de utilidades, ese experimento
      la omite con advertencia y continúa con las demás.

Salida:
    results/resumen_experimentos.json   — resultados consolidados de todos
    results/informe_experimentos.txt    — texto legible para el informe
    results/experimento_N_*.json        — datos detallados por experimento
    results/experimento_N_*.csv         — tablas planas para importar en Excel
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = {
    1: {
        "nombre": "Impacto del número de iteraciones Monte Carlo",
        "modulo": "experiments.experimento_1_iteraciones",
        "funcion": "run",
    },
    "1b": {
        "nombre": "Curvas de convergencia Monte Carlo (complemento Exp. 1)",
        "modulo": "experiments.experimento_1b_convergencia",
        "funcion": "run",
    },
    2: {
        "nombre": "Impacto de la temperatura en diversidad y utilidad",
        "modulo": "experiments.experimento_2_temperatura",
        "funcion": "run",
    },
    3: {
        "nombre": "Escalabilidad en instancias A, B, C",
        "modulo": "experiments.experimento_3_escalabilidad",
        "funcion": "run",
    },
    4: {
        "nombre": "Robustez: P(factibilidad) para distintas rutas",
        "modulo": "experiments.experimento_4_robustez",
        "funcion": "run",
    },
}

# IDs en el orden de ejecución recomendado
DEFAULT_ORDER = [1, "1b", 2, 3, 4]


def _import_and_run(exp_id) -> tuple[bool, Any, float]:
    import importlib
    meta   = EXPERIMENTS[exp_id]
    module = importlib.import_module(meta["modulo"])
    fn     = getattr(module, meta["funcion"])

    t0 = time.perf_counter()
    try:
        results = fn()
        return True, results, time.perf_counter() - t0
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"\n✗ Experimento {exp_id} falló: {exc}", file=sys.stderr)
        logging.exception("Experimento %s falló", exp_id)
        return False, str(exc), elapsed


def _build_informe(summary: dict) -> str:
    lines = [
        "=" * 70,
        "INFORME DE EXPERIMENTOS — SISTEMA DE RUTAS DE APRENDIZAJE",
        "=" * 70,
        f"Fecha de ejecución : {summary['timestamp']}",
        f"Experimentos       : {len(summary['ejecuciones'])} ejecutados",
        f"Exitosos / Fallidos: {summary['exitosos']} / {summary['fallidos']}",
        "",
    ]

    for ej in summary["ejecuciones"]:
        eid  = ej["experimento"]
        meta = EXPERIMENTS[eid]
        ok   = ej["exitoso"]
        lines += [
            "─" * 70,
            f"Experimento {eid}: {meta['nombre']}",
            f"  Estado  : {'✓ Completado' if ok else '✗ Fallido'}",
            f"  Tiempo  : {ej['tiempo_s']:.2f} s",
        ]

        if not ok:
            lines.append(f"  Error   : {ej.get('error', 'desconocido')}")
            continue

        res = ej.get("resultados", [])
        if not res:
            continue

        if eid == 1:
            lines.append("  n_iter  |  u_media  |  u_std  |  t_medio_s")
            for r in res:
                lines.append(
                    f"  {r['n_iter']:>6}  |  {r['utilidad_media']:>7.2f}  "
                    f"|  {r['utilidad_std']:>5.3f}  |  {r['tiempo_medio_s']:>10.4f}"
                )

        elif eid == "1b":
            lines.append("  Curvas de convergencia (u_media por checkpoint):")
            for curva in res[:3]:  # primera rep como ejemplo
                ck = curva.get("checkpoint")
                u  = curva.get("utilidad_mejor")
                if ck and u:
                    lines.append(f"    iter={ck:>5}  u={u:.2f}")

        elif eid == 2:
            lines.append("  temp  |  u_media  |  u_std  |  jaccard  |  diversidad")
            for r in res:
                lines.append(
                    f"  {r['temperatura']:>4.1f}  |  {r['utilidad_media']:>7.2f}  "
                    f"|  {r['utilidad_std']:>5.3f}  |  {r['jaccard_medio']:>7.4f}  "
                    f"|  {r['diversidad']:>9.4f}"
                )

        elif eid == 3:
            lines.append(
                "  inst  |  n  |  dp_u  |  dp_t_s  |  gr_u  |  gr_gap%  |  mc_u  |  mc_t_s"
            )
            for r in res:
                dp_u = f"{r['dp_utilidad']:.2f}" if r["dp_utilidad"] is not None else "N/A"
                dp_t = f"{r['dp_tiempo_s']:.4f}" if r["dp_tiempo_s"] is not None else "N/A"
                gr_g = f"{r['greedy_gap_pct']:.1f}" if r["greedy_gap_pct"] is not None else "N/A"
                lines.append(
                    f"  {r['instancia_label']:>4}  |  {r['n_cursos']:>2}  |  {dp_u:>6}  "
                    f"|  {dp_t:>8}  |  {r['greedy_utilidad']:>6.2f}  |  {gr_g:>7}  "
                    f"|  {r['mc_utilidad_media']:>6.2f}  |  {r['mc_tiempo_medio_s']:>7.4f}"
                )

        elif eid == 4:
            lines.append("  inst  |  ruta               |  cv  |  P(fact)  |  riesgo")
            prev = ""
            for r in res:
                key = f"{r['instancia_label']}/{r['ruta']}"
                sep = "  ···\n" if key != prev else ""
                prev = key
                lines.append(
                    f"{sep}  {r['instancia_label']:>4}  |  {r['ruta']:20s}  "
                    f"|  {r['cv']:.2f}  |  {r['p_factible_pct']:>6.2f}%  "
                    f"|  {r['nivel_riesgo']}"
                )

        lines.append("")

    lines += [
        "─" * 70,
        f"Tiempo total de experimentación: {summary['tiempo_total_s']:.2f} s",
        "=" * 70,
    ]
    return "\n".join(lines)


def main(exp_ids: list) -> None:
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'#'*60}")
    print(f"# SUITE DE EXPERIMENTOS — Rutas de Aprendizaje Óptimas")
    print(f"# {timestamp}")
    print(f"# Experimentos a ejecutar: {exp_ids}")
    print(f"{'#'*60}")

    ejecuciones: list[dict[str, Any]] = []
    t_suite_start = time.perf_counter()

    for exp_id in exp_ids:
        if exp_id not in EXPERIMENTS:
            print(f"\n⚠  Experimento {exp_id} no existe. Ignorando.")
            continue

        meta = EXPERIMENTS[exp_id]
        print(f"\n{'#'*60}")
        print(f"# Experimento {exp_id}: {meta['nombre']}")
        print(f"{'#'*60}")

        ok, results, elapsed = _import_and_run(exp_id)

        entry: dict[str, Any] = {
            "experimento": exp_id,
            "nombre":      meta["nombre"],
            "exitoso":     ok,
            "tiempo_s":    round(elapsed, 3),
        }
        if ok:
            entry["resultados"] = results
            entry["archivos"] = {
                "json": str(RESULTS_DIR / f"experimento_{exp_id}_*.json"),
                "csv":  str(RESULTS_DIR / f"experimento_{exp_id}_*.csv"),
            }
        else:
            entry["error"] = str(results)

        ejecuciones.append(entry)

    t_total = time.perf_counter() - t_suite_start

    summary: dict[str, Any] = {
        "timestamp":      timestamp,
        "experimentos":   [str(e) for e in exp_ids],
        "tiempo_total_s": round(t_total, 3),
        "exitosos":       sum(1 for e in ejecuciones if e["exitoso"]),
        "fallidos":       sum(1 for e in ejecuciones if not e["exitoso"]),
        "ejecuciones":    ejecuciones,
    }

    resumen_json = RESULTS_DIR / "resumen_experimentos.json"
    with resumen_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    informe_txt = RESULTS_DIR / "informe_experimentos.txt"
    informe = _build_informe(summary)
    with informe_txt.open("w", encoding="utf-8") as f:
        f.write(informe)

    print(f"\n\n{'#'*60}")
    print(f"# SUITE COMPLETADA en {t_total:.2f} s")
    print(f"# ✓ {summary['exitosos']} exitosos  |  ✗ {summary['fallidos']} fallidos")
    print(f"{'#'*60}")
    print(f"\nArchivos de resultados:")
    print(f"  {resumen_json}")
    print(f"  {informe_txt}")
    for path in sorted(RESULTS_DIR.glob("experimento_*.json")):
        csv = path.with_suffix(".csv")
        size_j = path.stat().st_size / 1024
        size_c = csv.stat().st_size / 1024 if csv.exists() else 0
        print(f"  {path.name:<45} {size_j:>5.1f} KB  |  "
              f"{csv.name:<45} {size_c:>5.1f} KB")

    print(f"\n{informe}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Runner de experimentos del sistema de rutas de aprendizaje."
    )
    parser.add_argument(
        "--exp",
        nargs="+",
        default=[str(e) for e in DEFAULT_ORDER],
        help="IDs de experimentos a ejecutar. Valores: 1 1b 2 3 4. Default: todos.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Muestra los experimentos disponibles y sale.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.list:
        print("\nExperimentos disponibles:")
        for eid, meta in EXPERIMENTS.items():
            print(f"  [{eid:>2}] {meta['nombre']}")
        sys.exit(0)

    # Convertir strings a int donde corresponda
    ids = []
    for e in args.exp:
        try:
            ids.append(int(e))
        except ValueError:
            ids.append(e)  # "1b" se queda como string

    main(ids)
