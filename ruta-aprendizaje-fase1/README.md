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
│   │   └── cursos.json              # Dataset base: 35 cursos (Fase 1)
│   └── instances/
│       ├── instancia_A_pequena.json # 10 cursos, T_max=80 h  — sanity check
│       ├── instancia_B_mediana.json # 22 cursos, T_max=200 h — evaluación intermedia
│       └── instancia_C_grande.json  # 35 cursos, T_max=300 h — estrés del optimizador
├── docs/
│   ├── fase1_modelado_formal.md     # Formalización matemática (Fase 1)
│   └── fase2_llm_integracion.md     # Diseño de prompts y LLM (Fase 2, WIP)
├── src/
│   ├── problem.py                   # Modelo formal: DAG, nodos, restricciones
│   ├── instance.py                  # Carga y validación de instancias JSON
│   ├── run_example.py               # Script de demostración rápida
│   ├── llm/
│   │   ├── client.py                # Wrapper de la API del LLM (Fase 2)
│   │   ├── prompts.py               # Ingeniería de prompts (Fase 2)
│   │   ├── models.py                # Modelos Pydantic de salida del LLM (Fase 2)
│   │   └── cache.py                 # Caché local de respuestas LLM (Fase 2)
│   └── solver/
│       ├── baseline.py              # Solver clásico DP sin LLM (Fase 3)
│       └── llm_assisted.py          # Solver híbrido con puntuación LLM (Fase 3)
└── tests/
    └── test_problem.py              # Pruebas unitarias del modelo de datos
```

---

## Estado del proyecto

| Fase | Contenido | Estado |
|------|-----------|--------|
| **Fase 1** | Modelado formal + Dataset + Instancias de prueba | ✅ Completada |
| **Fase 2** | Integración LLM + Prompts + Validación Pydantic | 🔄 En progreso |
| **Fase 3** | Algoritmo clásico + Solver híbrido | ⏳ Pendiente |
| **Fase 4** | Experimentos + Informe técnico | ⏳ Pendiente |

---

## Instalación

```bash
# Clonar el repositorio
git clone <URL_DEL_REPO>
cd ruta-aprendizaje

# Crear entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar credenciales LLM (Fase 2)
cp .env.example .env
# Editar .env con tu OPENAI_API_KEY
```

---

## Ejecución rápida (Fase 1)

```bash
# Verificar el dataset y las instancias de prueba
python src/run_example.py
```

La salida muestra:
- Validación del DAG (sin ciclos)
- Número de nodos, aristas y duración total
- Información de cada instancia de prueba (cursos incluidos, T_max, duración alcanzable)

---

## Dataset

El archivo `data/raw/cursos.json` contiene **35 cursos** del dominio
*"Ciencia de Datos e Inteligencia Artificial"* organizados en 7 niveles de profundidad
(L0–L6) que forman un **Grafo Dirigido Acíclico (DAG)** verificado.

- Duración por curso: entre 10 y 60 horas
- Duración total del catálogo: 1 295 horas
- Prerrequisitos: aristas del DAG, sin ciclos

Consulta `docs/fase1_modelado_formal.md` para la formalización matemática completa.

---

## Referencia rápida del modelo formal

El problema se modela como:

$$S^* = \underset{S \subseteq V}{\arg\max} \sum_{v \in S} u(v)$$

sujeto a:

$$\sum_{v \in S} d(v) \leq T_{\max}$$
$$\forall\, v_j \in S,\; \forall\, (v_i, v_j) \in E \;\Rightarrow\; v_i \in S$$

donde $u(v) \in [1,10]$ es la utilidad semántica asignada por el LLM y $d(v)$ es la
duración en horas del curso $v$.
