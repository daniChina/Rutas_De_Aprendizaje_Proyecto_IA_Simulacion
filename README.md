# Proyecto Final IA — Rutas de Aprendizaje Óptimas

Sistema híbrido que genera rutas de aprendizaje personalizadas combinando:
- un modelo formal de selección de cursos en un **DAG** con prerrequisitos,
- una evaluación semántica de relevancia con un **LLM**,
- un solver híbrido que compara **DP exacto / greedy / Monte Carlo**.

---

## ¿Qué hace este proyecto?

El objetivo es construir una ruta de cursos que maximice la utilidad total del alumno
respecto a un objetivo de aprendizaje en lenguaje natural, respetando:
- la duración máxima disponible `T_max`,
- la clausura de prerrequisitos del DAG.

Se trata de una variante del problema de la mochila con dependencias, donde el valor
semántico de cada curso no es fijo, sino que se infiere con un Modelo de Lenguaje.

---

## Estructura principal del proyecto

```text
.
├── data/
│   ├── raw/
│   │   └── cursos.json                  # Dataset base de 35 cursos
│   ├── instances/
│   │   ├── instancia_A_pequena.json     # 10 cursos, T_max=80 h
│   │   ├── instancia_B_mediana.json     # 22 cursos, T_max=200 h
│   │   └── instancia_C_grande.json      # 35 cursos, T_max=300 h
│   ├── output/                          # Salidas de pipeline.py
│   └── processed/                       # Instancias evaluadas por el LLM
├── docs/
│   ├── fase1_modelado_formal.md         # Formalización matemática
│   └── fase2_llm_integracion.md         # Diseño e ingeniería de prompts
├── src/
│   ├── problem.py                       # Modelo formal y validaciones DAG
│   ├── instance.py                      # Carga y serialización JSON
│   ├── run_example.py                   # Demo Fase 1
│   ├── run_fase2.py                     # Demo Fase 2 (evaluación LLM)
│   ├── run_fase2_batch.py               # Evaluación batch de instancias
│   ├── run_fase3.py                     # Solución híbrida para instancias reales
│   ├── llm/
│   │   ├── client.py                    # Cliente multi-proveedor LLM y reintentos
│   │   ├── prompts.py                   # Prompt engineering y schema JSON
│   │   ├── models.py                    # Validación de salida LLM con Pydantic
│   │   ├── evaluator.py                 # Orquestador de evaluación y guardado
│   │   └── cache.py                     # Caché local de respuestas LLM
│   └── solver/
│       ├── baseline.py                  # DP exacto/heurístico para knapsack con precedencia
│       ├── llm_assisted.py              # Solver híbrido comparativo DP/Greedy/MC
│       ├── mc_sampler.py                # Muestreo Monte Carlo de rutas válidas
│       └── robustness.py                # Análisis de robustez de rutas
├── tests/
│   ├── test_problem.py
│   ├── test_llm_fase2.py
│   ├── test_llm.py
│   ├── test_fallback.py
│   ├── test_fase3.py
│   └── test_simulacion.py
├── pipeline.py                          # Pipeline completo de entrada a salida
├── requirements.txt                     # Dependencias del proyecto
└── .env.example                         # Ejemplo de configuración LLM
```

---

## Estado de implementación

- **Fase 1**: completada — modelado formal del problema, dataset y validación del DAG.
- **Fase 2**: completada — evaluación semántica de cursos con LLM, prompts estructurados y caché.
- **Fase 3**: implementado — solver híbrido que ejecuta DP, greedy y Monte Carlo.
- **Experimentos**: scripts existentes, con espacio para ampliar análisis comparativo.

---

## Instalación

```bash
git clone <URL_DEL_REPO>
cd "c:\Computer Science Career\Computer Science Career\!!Third Year\IA\Ultimas modf\Rutas_De_Aprendizaje_Proyecto_IA_Simulacion"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Luego copia la configuración de entorno:

```bash
copy .env.example .env
```

Edita `.env` y define al menos:
- `LLM_PROVIDER` (por ejemplo `groq` o `openai`)
- `LLM_FALLBACK_PROVIDER`
- `OPENAI_API_KEY` o `GEMINI_API_KEY`
- `OPENAI_BASE_URL` si usas Groq

---

## Cómo ejecutar

### 1) Validar Fase 1

```bash
python src/run_example.py
```

Verifica el dataset completo y las instancias de prueba, valida el DAG y muestra estadísticas.

### 2) Evaluar instancias con el LLM (Fase 2)

```bash
python src/run_fase2.py
```

Carga la instancia mediana, evalúa cada curso con el LLM y guarda el dataset enriquecido en
`data/processed/`.

### 3) Ejecutar el solver híbrido (Fase 3)

```bash
python src/run_fase3.py
```

Resuelve las instancias evaluadas y guarda las rutas óptimas en `outputs/`.

### 4) Ejecutar pipeline completo

```bash
python pipeline.py --instancia data/instances/instancia_C_grande.json --objetivo "Quiero especializarme en NLP y LLMs"
```

Ejecuta carga, evaluación semántica y optimización en un único flujo.

### 5) Ejecutar tests

```bash
python -m pytest tests/ -v
```

---

## Casos de uso

- **Generación de rutas de aprendizaje personalizadas** para estudiantes que desean un plan estructurado con cursos relevantes a un objetivo profesional.
- **Evaluación semántica del catálogo** cuando el valor de los cursos depende del objetivo del alumno y no solo de métricas fijas.
- **Combinación de dependencias y presupuesto**: útil para elegir un conjunto de cursos que cumpla prerrequisitos y límite de horas.
- **Comparación de estrategias de optimización** entre soluciones exactas, heurísticas y Monte Carlo.
- **Prototipado de sistemas híbridos** que fusionan algoritmos clásicos de optimización con LLMs como oráculo de utilidad.

---

## Cómo funciona el problema

El problema se modela como:
- `V`: cursos del dataset.
- `E`: relaciones de prerrequisitos entre cursos.
- `T_max`: tiempo máximo disponible.
- `d(v)`: duración en horas de cada curso.
- `u(v)`: utilidad semántica inferida por el LLM.

La meta es seleccionar un subconjunto `S ⊆ V` que:
- cumpla `Σ d(v) ≤ T_max`,
- satisfaga la clausura de prerrequisitos (`si v_j ∈ S entonces sus prerrequisitos también`),
- maximice `Σ u(v)`.

Es una variante del problema de la mochila con precedencias en un grafo acíclico.

---

## Soluciones implementadas

### Fase 1 — Modelo formal y gestión de instancias

- `src/problem.py`: define `Course` y `LearningPathProblem`.
- `src/instance.py`: carga instancias JSON y serializa problemas.
- Validación del DAG, orden topológico, selección válida y métricas de utilidad/duración.

### Fase 2 — Evaluación semántica con LLM

- `src/llm/client.py`: cliente multi-proveedor con reintentos y fallback.
- `src/llm/prompts.py`: prompt engineering con ejemplos few-shot y esquema JSON.
- `src/llm/evaluator.py`: evalúa cada curso y guarda la instancia enriquecida.
- `src/llm/cache.py`: caché local para evitar llamadas LLM repetidas.

### Fase 3 — Solver híbrido

- `src/solver/baseline.py`:
  - DP exacta por máscara de bits para instancias pequeñas (n ≤ 20).
  - DP heurística para instancias más grandes.
- `src/solver/llm_assisted.py`:
  - orquesta DP, greedy y Monte Carlo.
  - obtiene la mejor solución según utilidad total.
- `src/solver/mc_sampler.py`: muestreo Monte Carlo de rutas factibles.
- `src/solver/robustness.py`: análisis de robustez ante variaciones de duración.

### Scripts de soporte

- `src/run_example.py`: demo de carga y validación de la Fase 1.
- `src/run_fase2.py`: demo de evaluación semántica del catálogo.
- `src/run_fase3.py`: demo de ejecución del solver híbrido en instancias reales.
- `pipeline.py`: pipeline completo de evaluación y optimización.

---

## Archivos de salida

- `data/processed/`: instancias evaluadas por el LLM.
- `outputs/`: rutas óptimas generadas por el solver híbrido.
- `data/output/`: resultados del pipeline completo.

---

## Configuración del LLM

- Copia `.env.example` a `.env`.
- Rellena la clave del proveedor elegido y los modelos opcionales.
- `LLM_PROVIDER` controla el proveedor activo.
- `LLM_FALLBACK_PROVIDER` define el proveedor de respaldo.
- `GEMINI_API_KEY` para Gemini.
- `OPENAI_API_KEY` para OpenAI o Groq.
- `OPENAI_BASE_URL` para Groq.

---

## Notas finales

Este repositorio integra razonamiento clásico de optimización con evaluación semántica de un LLM.
Los módulos están separados por fases para facilitar su extensión y validación.
