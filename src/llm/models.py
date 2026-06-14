"""
models.py
=========
Modelos Pydantic para validación de respuestas estructuradas del LLM.

Rol en el sistema:
  - EvaluacionCurso: contrato de datos entre la API del LLM y el evaluador.
    Garantiza que la respuesta cruda del modelo tiene el tipo, rango y campos
    correctos antes de que el dato llegue al objeto Course del problema.

Nota de diseño: este módulo solo define la RESPUESTA del LLM.
El objeto Course (src/problem.py) ya tiene los campos utilidad_relativa y
justificacion, por lo que no se necesita un DTO de "CursoEvaluado" separado.
El evaluador (src/llm/evaluator.py) aplica EvaluacionCurso directamente sobre
el Course usando course.utilidad_relativa y course.justificacion.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class EvaluacionCurso(BaseModel):
    """
    Representa la evaluación semántica de un curso individual devuelta por el LLM.

    El LLM debe producir exactamente este JSON para cada curso evaluado.
    Pydantic valida tipos, rangos y presencia de campos obligatorios antes
    de que el dato pase al componente clásico de optimización.
    """

    curso_id: str = Field(
        description="Identificador único del curso, tal como aparece en el dataset."
    )
    utilidad_relativa: int = Field(
        ge=1,
        le=10,
        description=(
            "Puntuación entera de utilidad semántica respecto al objetivo del usuario. "
            "Escala: 1 (irrelevante) → 10 (imprescindible)."
        ),
    )
    justificacion_breve: str = Field(
        min_length=20,
        max_length=600,
        description=(
            "Explicación concisa (1-3 oraciones) de por qué el curso tiene esa "
            "puntuación en relación al objetivo declarado por el usuario."
        ),
    )

    @field_validator("curso_id")
    @classmethod
    def curso_id_no_vacio(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("curso_id no puede ser una cadena vacía.")
        return v

    @field_validator("justificacion_breve")
    @classmethod
    def justificacion_no_vacia(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("justificacion_breve no puede estar vacía.")
        return v
