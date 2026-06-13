# Proyecto Final IA — Rutas de Aprendizaje Óptimas

Sistema híbrido que genera rutas de aprendizaje personalizadas combinando un
algoritmo clásico de optimización (knapsack con restricciones de precedencia en DAG)
con un Modelo de Lenguaje Grande (LLM) como evaluador semántico de utilidad.

---

## Estructura del proyecto

```
ruta-aprendizaje/
├── data/
│   ├── raw/
│   │   └── cursos.json                  # Dataset base: 35 cursos (Fase 1)
│   ├── instances/
│   │   ├── instancia_A_pequena.json     # 10 cursos, T_max=80 h  — sanity check
│   │   ├── instancia_B_mediana.json     # 22 cursos, T_max=200 h — evaluación intermedia
│   │   └── instancia_C_grande.json      # 35 cursos, T_max=300 h — estrés del optimizador
│   └── processed/                       # Instancias evaluadas por el LLM (Fase 2, generado)
├── docs/
│   ├── fase1_modelado_formal.md         # Formalización matemática (Fase 1)
│   └── fase2_llm_integracion.md         # Diseño de prompts y arquitectura LLM (Fase 2)
├── src/
│   ├── problem.py                       # Modelo formal: DAG, Course, LearningPathProblem
│   ├── instance.py                      # Carga y serialización de instancias JSON
│   ├── run_example.py                   # Demo Fase 1: carga instancias, valida DAG
│   ├── run_fase2.py                     # Demo Fase 2: evaluación LLM del catálogo
│   ├── llm/
│   │   ├── __init__.py                  # Exporta la interfaz pública del módulo
│   │   ├── client.py                    # LLMClient: wrapper OpenAI con reintentos
│   │   ├── prompts.py                   # System prompt (Few-Shot) + user prompt
│   │   ├── models.py                    # EvaluacionCurso (Pydantic)
│   │   ├── evaluator.py                 # evaluar_problema(), guardar_problema_evaluado()
│   │   └── cache.py                     # Caché local de respuestas LLM
│   └── solver/
│       ├── baseline.py                  # Solver DP clásico (Fase 3)
│       └── llm_assisted.py              # Solver híbrido LLM + DP (Fase 3)
└── tests/
    ├── test_problem.py                  # 11 tests — Fase 1 ✅
    └── test_llm_fase2.py                # 15 tests — Fase 2 ✅
```

---

## Estado del proyecto

| Fase | Contenido | Estado |
|------|-----------|--------|
| **Fase 1** | Modelado formal + Dataset (35 cursos) + Instancias A/B/C | ✅ Completada |
| **Fase 2** | LLM (Few-Shot, JSON mode, Pydantic, reintentos, caché) | ✅ Completada |
| **Fase 3** | Algoritmo DP clásico + Solver híbrido | ⏳ Pendiente |
| **Fase 4** | Experimentos + Informe técnico | ⏳ Pendiente |

---

## Instalación

```bash
git clone <URL_DEL_REPO>
cd ruta-aprendizaje

python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt

cp .env.example .env
# Editar .env y agregar OPENAI_API_KEY=sk-...
```

---

## Ejecución

### Fase 1 — Verificar dataset e instancias
```bash
python src/run_example.py
```
Valida el DAG de los 35 cursos, muestra el orden topológico y verifica las 3 instancias.

### Fase 2 — Evaluación semántica con el LLM
```bash
python src/run_fase2.py
```
Carga la Instancia A (10 cursos), llama al LLM para asignar `u(v) ∈ [1,10]` a cada
curso y guarda el resultado en `data/processed/instancia_A_evaluada.json`.

### Tests
```bash
# Todos los tests (sin pytest)
python tests/test_problem.py
python tests/test_llm_fase2.py

# Con pytest (si está instalado)
python -m pytest tests/ -v
```

---

## Arquitectura del sistema híbrido

```
cursos.json  (Fase 1)
     │
     ▼
LearningPathProblem          ← src/problem.py + src/instance.py
     │
     ▼  evaluar_problema()   ← src/llm/evaluator.py
LLMClient.evaluar_curso()    ← src/llm/client.py
  │  └─ construir_system_prompt()  ← Few-Shot + JSON Schema
  │  └─ construir_user_prompt()    ← objetivo + descripción del curso
  ▼
EvaluacionCurso (Pydantic)   ← src/llm/models.py  →  u(v) ∈ [1,10]
     │
     ▼
LearningPathProblem con u(v) → data/processed/*_evaluada.json
     │
     ▼  (Fase 3)
dp_knapsack_dag()            ← src/solver/baseline.py
     │
     ▼
S* ⊆ V  (ruta óptima)
```

---

## Referencia rápida del modelo formal

$$S^* = \underset{S \subseteq V}{\arg\max} \sum_{v \in S} u(v)$$

sujeto a $\sum_{v \in S} d(v) \leq T_{\max}$ y a la clausura de prerrequisitos del DAG.

donde $u(v) = \mathcal{F}_{\text{LLM}}(\text{desc}(v),\; \text{objetivo})$ y $d(v)$ es la duración en horas.
