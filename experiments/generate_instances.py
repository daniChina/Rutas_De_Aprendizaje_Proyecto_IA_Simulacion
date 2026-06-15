"""
generate_instances.py
=====================
Generador de instancias sintéticas para los experimentos.

Genera instancias A (pequeña), B (mediana) y C (grande) con:
  - Cursos con utilidades ya asignadas (simulando la salida de la Fase 2).
  - Duraciones realistas (entre 20 y 80 horas por curso).
  - Grafos DAG con densidad de aristas controlada.
  - T_max calibrado para que la solución óptima use entre el 60-80% del total.

Uso:
    python experiments/generate_instances.py

Salida:
    data/instances/instancia_A_pequena.json   (10 cursos)
    data/instances/instancia_B_mediana.json   (20 cursos)
    data/instances/instancia_C_grande.json    (35 cursos)
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

INSTANCES_DIR = ROOT / "data" / "instances"
SEED = 42


CATALOG = [
    # (id, titulo, descripcion, duracion_horas, utilidad_sugerida)
    ("CS_101", "Fundamentos de Programación", "Introducción a la programación con Python: variables, bucles, funciones y estructuras de datos básicas.", 30, 6),
    ("CS_102", "Álgebra Lineal para IA", "Vectores, matrices, transformaciones lineales, valores propios y descomposición SVD aplicados a ML.", 40, 7),
    ("CS_103", "Probabilidad y Estadística", "Variables aleatorias, distribuciones, inferencia bayesiana y tests de hipótesis.", 35, 7),
    ("CS_104", "Estructuras de Datos y Algoritmos", "Árboles, grafos, ordenamiento, búsqueda y análisis de complejidad.", 45, 5),
    ("CS_201", "Python Avanzado y Librerías Científicas", "NumPy, Pandas, Matplotlib y manejo eficiente de datos en Python.", 25, 8),
    ("CS_202", "Bases de Datos Relacionales", "SQL, modelado entidad-relación, normalización y consultas complejas.", 30, 4),
    ("CS_203", "Estadística Computacional", "Bootstrap, simulación Monte Carlo, MCMC y métodos de remuestreo.", 35, 6),
    ("CS_204", "Optimización Matemática", "Gradiente descendente, programación lineal, métodos de penalización y optimización convexa.", 40, 7),
    ("CS_301", "Machine Learning Clásico", "Regresión, clasificación, SVM, árboles de decisión, random forests y validación cruzada.", 50, 8),
    ("CS_302", "Redes Neuronales y Deep Learning", "Perceptrón, backpropagation, CNN, RNN y técnicas de regularización.", 55, 9),
    ("CS_303", "Procesamiento de Lenguaje Natural", "Tokenización, embeddings word2vec/GloVe, modelos de secuencia y análisis de sentimiento.", 45, 9),
    ("CS_304", "Visión por Computadora", "Convoluciones, detección de objetos, segmentación semántica y transfer learning en imágenes.", 50, 5),
    ("CS_305", "Aprendizaje por Refuerzo", "MDPs, Q-learning, policy gradients, actor-critic y entornos de simulación.", 55, 6),
    ("CS_306", "Modelos Generativos", "GANs, VAEs, flujos normalizadores y modelos de difusión.", 50, 7),
    ("CS_401", "Transformers y LLMs", "Atención multi-cabeza, BERT, GPT, T5 y fine-tuning eficiente con LoRA y QLoRA.", 60, 10),
    ("CS_402", "MLOps y Despliegue de Modelos", "Docker, Kubernetes, FastAPI, monitoreo de modelos y CI/CD para ML.", 40, 8),
    ("CS_403", "Big Data con Apache Spark", "RDDs, DataFrames, Spark SQL y pipelines de ML distribuido con MLlib.", 45, 5),
    ("CS_404", "Infraestructura Cloud para ML", "AWS/GCP/Azure: instancias GPU, almacenamiento, pipelines en la nube y costos.", 35, 6),
    ("CS_501", "Ingeniería de Prompts", "Técnicas avanzadas de prompting, few-shot, chain-of-thought y RAG.", 20, 8),
    ("CS_502", "Evaluación y Alineación de LLMs", "Benchmarks, RLHF, RLAIF, Constitutional AI y métricas de seguridad.", 35, 9),
    ("CS_503", "Recuperación de Información", "BM25, índices invertidos, búsqueda semántica densa y reranking.", 30, 7),
    ("CS_504", "Bases de Datos Vectoriales", "Faiss, Pinecone, Weaviate, búsqueda aproximada por vecinos y pipelines RAG.", 25, 8),
    ("CS_505", "Fine-tuning de Modelos de Lenguaje", "SFT, PEFT, LoRA, QLoRA, DPO y técnicas de alineación eficiente.", 40, 10),
    ("CS_506", "Sistemas Multi-Agente", "Frameworks de agentes LLM, herramientas, memoria y coordinación entre agentes.", 35, 9),
    ("CS_601", "Interpretabilidad y XAI", "SHAP, LIME, atención, sondas de activación y circuit-level interpretability.", 40, 7),
    ("CS_602", "Seguridad en Sistemas de IA", "Ataques adversariales, jailbreaking, red-teaming y defensa de modelos.", 35, 6),
    ("CS_603", "Ética en IA y Sesgo Algorítmico", "Fairness, accountability, transparencia y marcos legales de IA.", 25, 5),
    ("CS_604", "Computación Cuántica para ML", "Qubits, circuitos cuánticos, algoritmos cuánticos y NISQ para optimización.", 50, 3),
    ("CS_701", "Investigación en NLP", "Revisión de literatura, reproducibilidad, ablaciones y redacción de papers.", 45, 8),
    ("CS_702", "Sistemas de Recomendación", "Filtrado colaborativo, content-based, matrix factorization y LLMs para recomendación.", 40, 6),
    ("CS_703", "Detección de Anomalías", "Isolation Forest, autoencoders, one-class SVM y aplicaciones en seguridad.", 30, 5),
    ("CS_704", "Análisis de Grafos y GNNs", "PageRank, comunidades, GCN, GraphSAGE y aplicaciones en redes sociales.", 45, 6),
    ("CS_705", "Series Temporales y Forecasting", "ARIMA, Prophet, LSTM para temporales y Temporal Fusion Transformers.", 40, 5),
    ("CS_706", "Robótica e IA Embodied", "ROS, planificación de movimiento, SLAM y aprendizaje por imitación en robots.", 55, 4),
    ("CS_707", "Despliegue en Edge e IoT", "Cuantización, pruning, TensorFlow Lite, ONNX y modelos para dispositivos limitados.", 35, 5),
]

# Prerrequisitos reales del dominio
PREREQ_MAP: dict[str, list[str]] = {
    "CS_201": ["CS_101"],
    "CS_203": ["CS_103"],
    "CS_204": ["CS_102", "CS_103"],
    "CS_301": ["CS_201", "CS_203", "CS_204"],
    "CS_302": ["CS_301"],
    "CS_303": ["CS_302"],
    "CS_304": ["CS_302"],
    "CS_305": ["CS_301", "CS_204"],
    "CS_306": ["CS_302"],
    "CS_401": ["CS_303"],
    "CS_402": ["CS_301"],
    "CS_403": ["CS_201"],
    "CS_404": ["CS_402"],
    "CS_501": ["CS_401"],
    "CS_502": ["CS_401"],
    "CS_503": ["CS_303"],
    "CS_504": ["CS_503"],
    "CS_505": ["CS_401"],
    "CS_506": ["CS_501"],
    "CS_601": ["CS_302"],
    "CS_602": ["CS_302"],
    "CS_603": [],
    "CS_604": ["CS_204"],
    "CS_701": ["CS_303"],
    "CS_702": ["CS_301"],
    "CS_703": ["CS_301"],
    "CS_704": ["CS_302"],
    "CS_705": ["CS_302"],
    "CS_706": ["CS_305"],
    "CS_707": ["CS_302"],
}

INSTANCE_CONFIGS = {
    "A": {
        "instance_id": "instancia_A_pequena",
        "n_courses": 10,
        "ids": [
            "CS_101", "CS_102", "CS_103", "CS_201", "CS_203",
            "CS_204", "CS_301", "CS_302", "CS_303", "CS_401",
        ],
        "t_max_factor": 0.65,   # T_max = 65% de la duración total
    },
    "B": {
        "instance_id": "instancia_B_mediana",
        "n_courses": 20,
        "ids": [
            "CS_101", "CS_102", "CS_103", "CS_104", "CS_201",
            "CS_203", "CS_204", "CS_301", "CS_302", "CS_303",
            "CS_304", "CS_305", "CS_401", "CS_402", "CS_501",
            "CS_502", "CS_503", "CS_504", "CS_505", "CS_506",
        ],
        "t_max_factor": 0.60,
    },
    "C": {
        "instance_id": "instancia_C_grande",
        "n_courses": 35,
        "ids": [c[0] for c in CATALOG],   # todos
        "t_max_factor": 0.55,
    },
}


def build_instance(config: dict) -> dict[str, Any]:
    catalog_map = {c[0]: c for c in CATALOG}
    ids = config["ids"][: config["n_courses"]]

    cursos: list[dict[str, Any]] = []
    total_dur = 0

    for cid in ids:
        c = catalog_map[cid]
        prereqs = [p for p in PREREQ_MAP.get(cid, []) if p in ids]
        duracion = c[3]
        total_dur += duracion
        cursos.append({
            "id":                  cid,
            "titulo":              c[1],
            "descripcion":         c[2],
            "duracion_horas":      duracion,
            "prerrequisitos":      prereqs,
            "utilidad_relativa":   c[4],
            "justificacion_breve": (
                f"[SINTÉTICO] Puntuación {c[4]}/10 asignada por el diseño experimental. "
                f"Relevancia basada en el dominio de NLP y LLMs."
            ),
        })

    t_max = round(total_dur * config["t_max_factor"])

    return {
        "instance_id": config["instance_id"],
        "t_max":       t_max,
        "cursos":      cursos,
    }


def main() -> None:
    INSTANCES_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)

    print(f"\nGenerando instancias sintéticas en {INSTANCES_DIR}/")
    print("─" * 60)

    for label, config in INSTANCE_CONFIGS.items():
        data = build_instance(config)
        n     = len(data["cursos"])
        t_max = data["t_max"]
        total = sum(c["duracion_horas"] for c in data["cursos"])
        edges = sum(len(c["prerrequisitos"]) for c in data["cursos"])

        path = INSTANCES_DIR / f"{data['instance_id']}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(
            f"  [{label}] {data['instance_id']}\n"
            f"      Cursos={n} | Aristas={edges} | "
            f"Total={total}h | T_max={t_max}h ({t_max/total*100:.0f}%)\n"
            f"      → {path}"
        )

    print("\n✓ Instancias generadas correctamente.")
    print("  Próximo paso: python experiments/run_experiments.py")


if __name__ == "__main__":
    main()
