# Guía de testing — Rutas de Aprendizaje Óptimas

Cada sección cubre una fase del proyecto de forma independiente.
Los pasos están ordenados de menor a mayor dependencia: puedes
ejecutar Fase 1 sin API key, Fase 2 necesita credenciales, y
Fase 3 solo necesita que Fase 2 haya generado el archivo evaluado.

---

## Prerequisitos comunes

```bash
# Desde la raíz del proyecto
cd ruta-aprendizaje

python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

Verifica que el entorno quedó bien:

```bash
python -c "import pydantic, tenacity; print('Dependencias OK')"
```

---

## Fase 1 — Modelado formal y dataset

**Qué cubre:** el DAG de 35 cursos, la carga de instancias A/B/C,
la validación de prerrequisitos y el orden topológico.
No requiere API key ni conexión a internet.

### 1a. Demo visual completa

```bash
python src/run_example.py
```

Esperas ver tres bloques, uno por instancia, con el resumen del DAG
(`✓ DAG válido`), una selección de ejemplo y `Utilidad total`.
Al final: `✓ Fase 1 verificada.`

### 1b. Suite de tests unitarios (11 tests)

```bash
python -m pytest tests/test_problem.py -v
```

O sin pytest:

```bash
python tests/test_problem.py
```

Todos deben marcar `✓` o `PASSED`. Los tests cubren:

| Test | Qué verifica |
|---|---|
| `test_dag_valido_lineal` | Kahn detecta DAG correcto en grafo lineal |
| `test_dag_valido_ramificado` | Kahn en grafo con diamante A,B→C→D |
| `test_seleccion_valida_con_prerrequisitos` | `is_valid_selection([A, B])` = True |
| `test_seleccion_invalida_sin_prerrequisito` | `is_valid_selection([B])` = False |
| `test_seleccion_invalida_por_tiempo` | Rechaza cuando Σd > T_max |
| `test_seleccion_valida_exacta` | Acepta cuando Σd ≤ T_max justo |
| `test_orden_topologico_lineal` | A < B < C en posición |
| `test_orden_topologico_ramificado` | A,B < C < D |
| `test_utilidad_neutral_sin_llm` | `utility = 5.0` cuando `utilidad_relativa = None` |
| `test_valor_objetivo` | `objective_value([A,B])` = 10.0 con u=5 neutral |
| `test_dataset_completo_es_dag` | Los 35 cursos reales no tienen ciclos |

### Qué hacer si falla

- `FileNotFoundError: data/raw/cursos.json` → estás ejecutando desde
  la carpeta incorrecta. Usa `cd ruta-aprendizaje` primero.
- `ModuleNotFoundError: src` → falta el `sys.path.insert` o no usas
  `python -m pytest` desde la raíz.

---

## Fase 2 — Evaluación semántica con el LLM

**Qué cubre:** el cliente LLM (OpenAI / Gemini / Groq), los prompts
Few-Shot, la validación Pydantic, la caché local y el evaluador.

### 2a. Tests sin API (mocks) — 15 tests

No necesitan credenciales. Prueban la lógica interna con objetos mock:

```bash
python -m pytest tests/test_llm_fase2.py -v
```

O sin pytest:

```bash
python tests/test_llm_fase2.py
```

### 2b. Tests de resiliencia / fallback — sin API

Verifican que el cliente maneja errores de red, respuestas malformadas
y el mecanismo de reintentos con `tenacity`:

```bash
python -m pytest tests/test_fallback.py -v
```

O sin pytest:

```bash
python tests/test_fallback.py
```

### 2c. Test de conexión real — requiere API key

Primero configura `.env`:

```bash
cp .env.example .env
# Editar .env: poner LLM_PROVIDER, clave y modelo
```

Luego verifica la conexión con un único curso de prueba (no consume
el catálogo real):

```bash
python tests/test_llm.py
```

Salida esperada:

```
✓ Conexión exitosa. Respuesta recibida y validada:
  curso_id           : TEST_001
  utilidad_relativa  : 8/10
  justificacion      : …
```

### 2d. Demo completa de evaluación — requiere API key

Evalúa los 10 cursos de la Instancia A y guarda el resultado en
`data/processed/instancia_A_evaluada.json`:

```bash
python src/run_fase2.py
```

Esperas ver un ranking de utilidades con los 5 cursos más útiles
para el objetivo predefinido (NLP / LLMs) y la ruta del archivo guardado.

> **Nota sobre caché:** las respuestas del LLM se guardan en
> `.llm_cache.json`. Si cambias de proveedor, borra el archivo:
> `rm .llm_cache.json` (Linux/macOS) o `del .llm_cache.json` (Windows).

### Qué hacer si falla

- `EnvironmentError: OPENAI_API_KEY no configurada` → revisa `.env`
  y que el archivo esté en la raíz del proyecto.
- `ValidationError` de Pydantic → el modelo LLM devolvió JSON
  malformado. Prueba con `LLM_MODEL=gpt-4o-mini` o `gemini-2.5-flash`.
- Respuesta `None` en test_llm → límite de rate alcanzado; espera
  un minuto o cambia a Groq (14 400 peticiones/día gratis).

---

## Fase 3 — Solvers (Monte Carlo, DP exacto, greedy)

**Qué cubre:** los tres solvers y el análisis de robustez estocástica.
No requiere API key si la instancia ya está evaluada.

### 3a. Suite de tests Monte Carlo + robustez — sin API

```bash
python -m pytest tests/test_simulacion.py -v
```

O sin pytest:

```bash
python tests/test_simulacion.py
```

Los tests usan un problema sintético de 3 nodos con utilidades fijas,
así que no dependen del LLM.

### 3b. Test manual del solver DP exacto — sin API

Puedes probarlo directamente en un REPL o script. El siguiente bloque
usa la Instancia A con utilidades dummy (sin LLM):

```python
import sys; sys.path.insert(0, ".")
from src.instance import load_instance
from src.solver.baseline import dp_knapsack_dag, greedy_by_utility_density

problem = load_instance("data/instances/instancia_A_pequena.json")
# Asignar utilidades de ejemplo (normalmente las asigna el LLM en Fase 2)
for c in problem.courses:
    c.utilidad_relativa = 5

dp = dp_knapsack_dag(problem)
print(dp.summary())
print("Válida:", problem.is_valid_selection(dp.selected_ids))

g = greedy_by_utility_density(problem)
print(g.summary())
```

Salida esperada (con u=5 uniformes):

```
DP Solver — N cursos seleccionados
  Utilidad total  : XX.0
  Duracion total  : XX h
  ...
Válida: True
```

### 3c. Test manual con instancia evaluada por LLM

Requiere haber ejecutado el paso 2d antes.

```python
import sys; sys.path.insert(0, ".")
from src.instance import load_instance
from src.solver.baseline import dp_knapsack_dag, greedy_by_utility_density
from src.solver.mc_sampler import mc_path_sampler

problem = load_instance("data/processed/instancia_A_evaluada.json")

dp      = dp_knapsack_dag(problem)
greedy  = greedy_by_utility_density(problem)
mc      = mc_path_sampler(problem, n_iter=2000, temperature=1.0, seed=42)

print("=== DP exacto ===");    print(dp.summary())
print("=== Greedy    ===");    print(greedy.summary())
print("=== Monte Carlo ===");  print(mc.summary())
```

### 3d. Análisis de robustez estocástica

```python
from src.solver.robustness import robustness_analysis

rob = robustness_analysis(
    problem,
    mc.selected_ids,
    n_simulations=5000,
    cv=0.20,           # 20 % de variabilidad en las duraciones
)
print(f"P(ruta factible) : {rob.p_feasible:.1%}")
print(f"Duración media   : {rob.mean_simulated_duration:.0f} h")
print(f"Duración p95     : {rob.p95_duration:.0f} h")
```

### 3e. Solver híbrido completo (LLM + DP + MC)

Requiere API key y la instancia A evaluada.

```python
import sys; sys.path.insert(0, ".")
from src.instance import load_instance
from src.solver.llm_assisted import llm_assisted_solver

problem = load_instance("data/processed/instancia_A_evaluada.json")

result = llm_assisted_solver(
    problem,
    objetivo_usuario="Quiero aprender machine learning desde cero.",
    mc_iterations=2000,
    mc_seed=42,
)
print(result.summary())
print("IDs seleccionados:", result.selected_ids)
print("Solver ganador   :", result.solver_used)
```

### Qué hacer si falla

- `NotImplementedError` en `dp_knapsack_dag` → reemplaza `baseline.py`
  con la versión implementada en esta entrega.
- `FileNotFoundError: instancia_A_evaluada.json` → ejecuta primero
  `python src/run_fase2.py` (paso 2d) para generar el archivo.
- Resultados del DP con selección vacía → revisa que los cursos tienen
  `utilidad_relativa` distinto de `None` antes de llamar al solver.

---

## Suite completa (todas las fases sin API)

```bash
python -m pytest tests/ -v --ignore=tests/test_llm.py
```

Ejecuta los 41 tests de las Fases 1, 2 (mocks) y 3 sin necesitar
credenciales. Todos deben terminar en `PASSED`.

## Suite completa incluyendo conexión real

```bash
python -m pytest tests/ -v
```

Requiere `.env` configurado con una API key válida.

---

## Referencia rápida de comandos

| Qué probar | Comando | API key |
|---|---|---|
| DAG y dataset | `python src/run_example.py` | No |
| Tests Fase 1 | `pytest tests/test_problem.py -v` | No |
| Tests Fase 2 (mocks) | `pytest tests/test_llm_fase2.py -v` | No |
| Tests fallback LLM | `pytest tests/test_fallback.py -v` | No |
| Conexión LLM real | `python tests/test_llm.py` | **Sí** |
| Evaluación semántica | `python src/run_fase2.py` | **Sí** |
| Tests Monte Carlo | `pytest tests/test_simulacion.py -v` | No |
| DP / greedy manual | REPL con `baseline.py` | No |
| Solver híbrido | REPL con `llm_assisted_solver()` | **Sí** |
| Todo sin API | `pytest tests/ -v --ignore=tests/test_llm.py` | No |
| Todo con API | `pytest tests/ -v` | **Sí** |
