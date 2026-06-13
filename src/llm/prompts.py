"""
prompts.py  (src/llm/prompts.py)
=================================
Ingeniería de prompts para la evaluación semántica de cursos.

Decisiones de diseño:
  - Se usa la estrategia Few-Shot con 3 ejemplos calibrados en el system prompt,
    uno por cada zona de la escala (baja, media, alta utilidad).
  - El formato de salida se especifica como JSON Schema embebido en el prompt,
    reforzando la instrucción de structured outputs de la API.
  - Se separa el system prompt (rol + instrucciones invariantes) del user prompt
    (datos variables por llamada) para aprovechar el caché de prefijo de la API
    y reducir costos en evaluaciones masivas.
"""

from __future__ import annotations

import json


# ---------------------------------------------------------------------------
# Schema de salida esperado (embebido en el prompt para refuerzo explícito)
# ---------------------------------------------------------------------------

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "curso_id": {
            "type": "string",
            "description": "ID exacto del curso tal como fue proporcionado."
        },
        "utilidad_relativa": {
            "type": "integer",
            "minimum": 1,
            "maximum": 10,
            "description": "Puntuación entera de utilidad semántica."
        },
        "justificacion_breve": {
            "type": "string",
            "description": "Justificación concisa de 1 a 3 oraciones."
        }
    },
    "required": ["curso_id", "utilidad_relativa", "justificacion_breve"],
    "additionalProperties": False
}


# ---------------------------------------------------------------------------
# Ejemplos Few-Shot calibrados
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = [
    # --- Ejemplo 1: Alta relevancia (puntuación 9) ---
    {
        "objetivo": (
            "Quiero convertirme en ingeniero de machine learning especializado "
            "en NLP y grandes modelos de lenguaje para trabajar en startups de IA."
        ),
        "curso": {
            "id": "CS_601",
            "titulo": "Transformers y Modelos de Lenguaje Grande (LLMs)",
            "descripcion": (
                "Estudia en profundidad la arquitectura Transformer: atención "
                "multi-cabeza, bloques encoder y decoder. Se analiza el proceso "
                "de preentrenamiento y el ajuste fino de modelos como BERT, GPT "
                "y T5. El estudiante aprende técnicas de eficiencia como LoRA, "
                "QLoRA y PEFT para adaptar LLMs con recursos limitados."
            )
        },
        "respuesta": {
            "curso_id": "CS_601",
            "utilidad_relativa": 9,
            "justificacion_breve": (
                "El curso cubre exactamente las arquitecturas Transformer y LLMs "
                "que son el núcleo del objetivo del usuario. Las técnicas de "
                "ajuste fino eficiente (LoRA, QLoRA) son habilidades directamente "
                "demandadas en startups de IA, haciendo este curso casi "
                "imprescindible para el perfil buscado."
            )
        }
    },
    # --- Ejemplo 2: Relevancia media (puntuación 5) ---
    {
        "objetivo": (
            "Quiero convertirme en ingeniero de machine learning especializado "
            "en NLP y grandes modelos de lenguaje para trabajar en startups de IA."
        ),
        "curso": {
            "id": "CS_403",
            "titulo": "Big Data con Apache Spark",
            "descripcion": (
                "Presenta el paradigma de procesamiento distribuido con Apache "
                "Spark: RDDs, DataFrames y Spark SQL. El estudiante aprende a "
                "procesar datasets de terabytes en clústeres y a implementar "
                "pipelines de ML distribuido con Spark MLlib."
            )
        },
        "respuesta": {
            "curso_id": "CS_403",
            "utilidad_relativa": 5,
            "justificacion_breve": (
                "Spark es una habilidad de infraestructura complementaria útil "
                "para procesar grandes corpora de texto, pero no es central en "
                "el rol de ingeniería de NLP/LLMs. Puede ser valioso en startups "
                "con grandes volúmenes de datos, pero no es prioritario frente a "
                "cursos de modelado directo."
            )
        }
    },
    # --- Ejemplo 3: Baja relevancia (puntuación 2) ---
    {
        "objetivo": (
            "Quiero convertirme en ingeniero de machine learning especializado "
            "en NLP y grandes modelos de lenguaje para trabajar en startups de IA."
        ),
        "curso": {
            "id": "CS_502",
            "titulo": "Visión por Computadora con Deep Learning",
            "descripcion": (
                "Profundiza en la visión computacional moderna: convoluciones, "
                "arquitecturas CNN emblemáticas y tareas de detección de objetos "
                "y segmentación semántica. Incluye transfer learning desde "
                "modelos preentrenados en ImageNet."
            )
        },
        "respuesta": {
            "curso_id": "CS_502",
            "utilidad_relativa": 2,
            "justificacion_breve": (
                "La visión por computadora es un dominio distinto al NLP/LLMs "
                "que busca el usuario. Aunque comparte fundamentos de deep "
                "learning, sus técnicas específicas (CNN para imágenes, "
                "detección de objetos) tienen escasa transferencia directa al "
                "objetivo declarado de trabajar con modelos de lenguaje."
            )
        }
    }
]


# ---------------------------------------------------------------------------
# Construcción del System Prompt
# ---------------------------------------------------------------------------

def construir_system_prompt() -> str:
    """
    Genera el system prompt completo con instrucciones de rol,
    criterios de evaluación, ejemplos few-shot y formato de salida.

    El system prompt es invariante entre llamadas (solo cambia el user prompt),
    lo que permite cachear el prefijo en la API y reducir costos.
    """

    ejemplos_formateados = "\n\n".join([
        f"### Ejemplo {i+1}\n"
        f"**Objetivo del usuario:** {ej['objetivo']}\n\n"
        f"**Curso a evaluar:**\n```json\n{json.dumps(ej['curso'], ensure_ascii=False, indent=2)}\n```\n\n"
        f"**Respuesta correcta:**\n```json\n{json.dumps(ej['respuesta'], ensure_ascii=False, indent=2)}\n```"
        for i, ej in enumerate(FEW_SHOT_EXAMPLES)
    ])

    schema_str = json.dumps(OUTPUT_SCHEMA, ensure_ascii=False, indent=2)

    return f"""Eres un experto en diseño curricular y evaluación pedagógica con experiencia \
en ciencia de datos e inteligencia artificial. Tu única tarea es analizar la relevancia \
semántica de un curso respecto al objetivo de aprendizaje declarado por un usuario y \
asignarle una puntuación de utilidad.

## Criterios de Evaluación

Asigna una puntuación de utilidad del 1 al 10 basándote en los siguientes criterios:

| Rango | Interpretación |
|-------|---------------|
| 9-10  | El curso es directamente esencial para el objetivo. Aborda las habilidades o conocimientos \
nucleares que el usuario necesita. |
| 7-8   | El curso es muy relevante y aporta competencias importantes para el objetivo, aunque no sea \
el foco central. |
| 5-6   | El curso es moderadamente útil: proporciona fundamentos necesarios o habilidades \
complementarias, pero su contribución es indirecta. |
| 3-4   | El curso tiene valor formativo general pero su conexión con el objetivo es débil o periférica. |
| 1-2   | El curso es irrelevante o está orientado a un dominio distinto al que persigue el usuario. |

## Instrucciones Críticas de Formato

- DEBES responder ÚNICAMENTE con un objeto JSON válido. Sin texto introductorio, sin \
explicaciones fuera del JSON, sin bloques de código markdown.
- El JSON debe ajustarse EXACTAMENTE al siguiente schema:

{schema_str}

- Si recibes un curso cuyo ID no reconoces, usa el ID tal como te fue proporcionado.
- La justificacion_breve debe ser concisa (1-3 oraciones) y referenciar explícitamente \
elementos del objetivo del usuario y de la descripción del curso.

## Ejemplos de Referencia (Few-Shot)

Los siguientes ejemplos ilustran el nivel de análisis y formato esperados. \
Utilízalos como calibración de tu escala:

{ejemplos_formateados}
"""


# ---------------------------------------------------------------------------
# Construcción del User Prompt (variable por llamada)
# ---------------------------------------------------------------------------

def construir_user_prompt(objetivo_usuario: str, curso: dict) -> str:
    """
    Genera el user prompt para una llamada específica.
    Solo contiene los datos variables: objetivo y curso a evaluar.

    Args:
        objetivo_usuario: Meta de aprendizaje expresada en lenguaje natural.
        curso: Diccionario con al menos 'id', 'titulo' y 'descripcion'.

    Returns:
        String con el user prompt formateado.
    """

    curso_resumido = {
        "id": curso.get("id", "DESCONOCIDO"),
        "titulo": curso.get("titulo", ""),
        "descripcion": curso.get("descripcion", "")
    }

    return (
        f"**Objetivo del usuario:**\n{objetivo_usuario.strip()}\n\n"
        f"**Curso a evaluar:**\n"
        f"{json.dumps(curso_resumido, ensure_ascii=False, indent=2)}\n\n"
        "Responde únicamente con el objeto JSON de evaluación."
    )
