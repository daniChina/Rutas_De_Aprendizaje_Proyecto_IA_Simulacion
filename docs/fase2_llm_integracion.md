# Fase 2: Integración y Configuración del LLM

## 2.1 Rol Funcional del LLM en el Sistema Híbrido

En la arquitectura propuesta, el **Modelo de Lenguaje Grande (LLM)** no actúa como
tomador de decisiones finales ni como optimizador de rutas. Su rol es más específico y
acotado: funcionar como un **oráculo de utilidad semántica**. Dado que el algoritmo
clásico de optimización opera exclusivamente sobre valores numéricos, necesita una señal
cuantitativa que capture la relevancia de cada curso respecto al perfil particular del
aprendiz. Esa señal no puede derivarse de los metadatos estructurados del dataset (el ID,
la duración o los prerrequisitos son agnósticos al objetivo del usuario), sino que requiere
comprensión del lenguaje natural.

El LLM resuelve exactamente ese problema. Para cada curso $v \in V$ del catálogo, el
modelo lee su descripción en lenguaje natural y la compara semánticamente con la meta de
aprendizaje declarada por el usuario, produciendo la función de utilidad:

$$u(v) = \mathcal{F}_{\text{LLM}}\bigl(\text{desc}(v),\; \text{objetivo del usuario}\bigr) \in [1, 10]$$

Este valor es el único puente entre el razonamiento cualitativo del LLM y el razonamiento
cuantitativo del optimizador. La separación de responsabilidades es deliberada: el LLM
evalúa relevancia semántica, y el algoritmo clásico resuelve la combinatoria de selección
bajo restricciones.

### 2.1.1 Proceso de Evaluación por Etapas

El flujo de evaluación sigue el siguiente pipeline por cada curso del catálogo:

```
Dataset JSON (cursos.json)
        │
        ▼
┌───────────────────┐
│  Construcción del │  ← System Prompt (invariante, cacheado)
│  Prompt (Few-Shot)│  ← User Prompt (variable: objetivo + desc. del curso)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   API del LLM     │  ← JSON mode (response_format={"type":"json_object"})
│   (OpenAI GPT-4o) │  ← temperatura=0.1 para respuestas deterministas
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Parseo + Pydantic│  ← json.loads() → EvaluacionCurso.model_validate()
│  Validación       │  ← Rango [1,10], campos obligatorios, tipos
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  CursoEvaluado    │  ← Objeto enriquecido con utilidad_relativa
│  (objeto Python)  │
└────────┬──────────┘
         │
         ▼
 cursos_evaluados.json   →   Fase 3: Algoritmo Clásico de Optimización
```

---

## 2.2 Estrategia de Ingeniería de Prompts

### 2.2.1 Estructura de Dos Capas: System Prompt y User Prompt

El diseño separa las instrucciones en dos capas con funciones distintas:

**System Prompt (invariante):** Define el rol del modelo ("experto en diseño
curricular"), los criterios de evaluación en forma de tabla semántica, el schema JSON
de salida y los ejemplos de calibración (Few-Shot). Esta capa se mantiene constante en
todas las llamadas del catálogo, lo que permite aprovechar el **caché de prefijo** de la
API de OpenAI (Prompt Caching), reduciendo tanto la latencia como el costo por token
en evaluaciones masivas.

**User Prompt (variable):** Contiene únicamente los datos específicos de cada llamada:
el objetivo del usuario y el JSON del curso a evaluar. Su extensión es mínima y varía
solo en el contenido del curso.

Esta separación no es un detalle de implementación menor: tiene consecuencias directas
sobre la consistencia de las evaluaciones. Al mantener el contexto de referencia (role,
criterios, ejemplos) siempre en la misma posición del contexto del modelo, se reduce la
variabilidad de las puntuaciones entre cursos.

### 2.2.2 Técnica Few-Shot para Calibración de Escala

Se incluyen **tres ejemplos anotados** en el system prompt, uno por zona semántica de
la escala (alta relevancia ≥ 9, relevancia media ≈ 5, baja relevancia ≤ 2). Los tres
ejemplos comparten el mismo objetivo de usuario (NLP / LLMs) para demostrar al modelo
cómo varía la puntuación según el dominio del curso, no según el objetivo.

Los ejemplos cumplen tres funciones simultáneas:

1. **Calibración de la escala:** evitan que el modelo use solo los extremos (1 o 10) o
   se concentre en el centro (todo 5). Los ejemplos muestran variación real.
2. **Anclaje del formato:** el modelo ve el JSON esperado en tres instancias antes de
   producir su propia respuesta, reduciendo errores de formato.
3. **Calibración semántica:** los ejemplos enseñan al modelo que la relevancia depende
   de la distancia entre dominios, no de la calidad intrínseca del curso.

### 2.2.3 Temperatura Baja y JSON Mode

Se configura `temperatura=0.1` en todas las llamadas. Los LLMs generativos, con
temperaturas altas, producen evaluaciones que varían entre ejecuciones para el mismo
par (objetivo, curso), lo que haría el sistema no reproducible. Con temperatura baja,
la evaluación se vuelve **casi determinista**, propiedad esencial para un componente
que alimenta un algoritmo de optimización: el mismo dataset debe producir la misma
ruta óptima en ejecuciones sucesivas.

El parámetro `response_format={"type": "json_object"}` activa el **JSON mode** de la
API, que garantiza sintácticamente que la respuesta es JSON parseable, incluso si el
contenido semántico fuera incorrecto. Esto elimina la clase de errores más frecuente
en la integración LLM → sistema clásico: respuestas con prefijos en lenguaje natural
("¡Claro! Aquí tienes la evaluación: {...}") que rompen el parseo.

---

## 2.3 Garantía de Robustez: Validación con Pydantic

La validación con Pydantic constituye la **barrera de calidad** entre el componente
no determinista (LLM) y el componente determinista (algoritmo de optimización). Aunque
el JSON mode garantiza JSON sintácticamente válido, no garantiza corrección semántica.
El modelo podría devolver `"utilidad_relativa": "nueve"` (string en lugar de int) o
`"utilidad_relativa": 15` (fuera de rango).

El modelo `EvaluacionCurso` de Pydantic v2 captura exactamente estas anomalías:

```python
class EvaluacionCurso(BaseModel):
    curso_id: str                    # tipo: string no vacío
    utilidad_relativa: int = Field(ge=1, le=10)  # rango: [1, 10]
    justificacion_breve: str = Field(min_length=20, max_length=300)
```

Si la validación falla, el sistema registra el error detallado y activa un **mecanismo
de fallback**: asigna al curso una puntuación neutra de 5/10 y continúa con el catálogo.
Esto garantiza que un fallo puntual en la API o una respuesta malformada no bloquee la
evaluación completa del catálogo.

---

## 2.4 Manejo de Errores y Resiliencia Operacional

El módulo implementa dos niveles de resiliencia:

**Nivel 1 — Reintentos automáticos con backoff exponencial (Tenacity):**
Los errores de red (`APIConnectionError`) y de cuota (`RateLimitError`) se reintentan
automáticamente hasta 4 veces, con esperas de 2, 4, 8 y 16 segundos
respectivamente. Solo errores recuperables activan el reintento; errores de
autenticación o de contenido se propagan inmediatamente para no consumir cuota.

**Nivel 2 — Fallback de puntuación neutra:**
Si todos los reintentos fallan, el curso recibe una puntuación de fallback (5/10 por
defecto) y el campo `evaluacion_exitosa` se marca como `False`. El algoritmo de
optimización puede distinguir estos casos para un post-análisis.

La siguiente tabla resume la taxonomía de errores y sus respuestas:

| Tipo de Error | Causa | Respuesta del Sistema |
|---|---|---|
| `APIConnectionError` | Fallo de red / timeout | Reintento con backoff × 4 |
| `RateLimitError` | Cuota de tokens agotada | Reintento con backoff × 4 |
| `APIError` (4xx/5xx) | Error del servidor OpenAI | Log + fallback inmediato |
| `json.JSONDecodeError` | Respuesta no es JSON válido | Log + fallback inmediato |
| `ValidationError` (Pydantic) | JSON válido pero estructura incorrecta | Log detallado + fallback |
| `ValidationError` (dataset) | Curso malformado en cursos.json | Skip del curso + continúa |

---

## 2.5 Estructura de Módulos y Archivos

```
fase2/
├── .env.example          # Plantilla de variables de entorno (no subir .env real)
├── models.py             # Modelos Pydantic: EvaluacionCurso, CursoEvaluado
├── prompts.py            # System prompt, user prompt y ejemplos few-shot
├── llm_evaluator.py      # Lógica de integración: evaluar_catalogo(), main()
├── cursos.json           # Dataset original de la Fase 1 (entrada)
└── cursos_evaluados.json # Dataset enriquecido con utilidades (salida → Fase 3)
```

---

## 2.6 Configuración y Ejecución

### Instalación de dependencias

```bash
pip install openai pydantic python-dotenv tenacity
```

### Configuración de credenciales

```bash
cp .env.example .env
# Editar .env y agregar OPENAI_API_KEY=sk-...
```

### Ejecución

```bash
python llm_evaluator.py
```

El script genera `cursos_evaluados.json` con todos los cursos del catálogo enriquecidos
con los campos `utilidad_relativa`, `justificacion_breve` y `evaluacion_exitosa`, listo
para ser consumido por el algoritmo clásico de la Fase 3.

### Uso programático (integración con Fase 3)

```python
import json
from llm_evaluator import evaluar_catalogo, guardar_dataset_enriquecido

with open("cursos.json", encoding="utf-8") as f:
    dataset = json.load(f)

objetivo = "Quiero especializarme en visión por computadora para sistemas autónomos."

cursos_evaluados = evaluar_catalogo(
    dataset_json=dataset,
    objetivo_usuario=objetivo,
    delay_entre_llamadas=0.5,   # Respetar rate limits
    puntuacion_fallback=5,      # Puntuación neutra ante fallos
)

guardar_dataset_enriquecido(cursos_evaluados, "cursos_evaluados.json")
```

---

## 2.7 Ejemplo de Salida del Dataset Enriquecido

El archivo `cursos_evaluados.json` tiene la misma estructura que el dataset original
más los tres campos añadidos por el LLM:

```json
{
  "id": "CS_502",
  "titulo": "Visión por Computadora con Deep Learning",
  "descripcion": "...",
  "duracion_horas": 45,
  "prerrequisitos": ["CS_402"],
  "utilidad_relativa": 9,
  "justificacion_breve": "El curso aborda directamente las arquitecturas CNN y
    técnicas de detección empleadas en sistemas autónomos, siendo esencial para
    el objetivo declarado por el usuario.",
  "evaluacion_exitosa": true
}
```

Los campos `utilidad_relativa` y `duracion_horas` son los dos valores numéricos que
consume el algoritmo de optimización en la Fase 3 para resolver el problema:

$$S^* = \underset{S \subseteq V}{\arg\max} \sum_{v \in S} u(v) \quad \text{s.a.} \quad \sum_{v \in S} d(v) \leq T_{\max}$$
