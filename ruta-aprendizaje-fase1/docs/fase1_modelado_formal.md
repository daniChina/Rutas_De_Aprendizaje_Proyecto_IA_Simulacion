# Fase 1: Definición del Modelo Formal y Diseño del Dataset

---

## 1. Modelado Formal del Problema

### 1.1 Definición del Grafo de Conocimiento

El dominio de aprendizaje se modela como un **Grafo Dirigido Acíclico (DAG)**:

$$G = (V, E)$$

donde:

- $V = \{v_1, v_2, \dots, v_n\}$ es el conjunto finito de **nodos**, cada uno representando un curso o recurso de aprendizaje dentro de un dominio de conocimiento coherente.

- $E \subseteq V \times V$ es el conjunto de **aristas dirigidas**, donde una arista $(v_i, v_j) \in E$ indica que el curso $v_i$ es **prerrequisito** del curso $v_j$; es decir, $v_j$ no puede ser incluido en una ruta de aprendizaje válida sin que $v_i$ haya sido completado previamente.

**Condición de aciclicidad:** El grafo satisface la propiedad DAG, garantizando que no existen ciclos en las dependencias:

$$\nexists \; (v_{i_1}, v_{i_2}, \dots, v_{i_k}) \in V^k \;\text{ tal que }\; (v_{i_1}, v_{i_2}), (v_{i_2}, v_{i_3}), \dots, (v_{i_k}, v_{i_1}) \in E$$

Esta condición es computacionalmente verificable mediante un recorrido topológico (DFS o Kahn's algorithm) sobre $G$.

---

### 1.2 Atributos de los Nodos

Cada nodo $v \in V$ posee dos atributos escalares que determinan su rol en el proceso de optimización:

**Duración:**

$$d: V \to \mathbb{R}^+ \qquad d(v) \in [d_{\min},\, d_{\max}]$$

Representa el tiempo de dedicación, medido en horas, requerido para completar el curso $v$. En el dataset construido, $d_{\min} = 10$ y $d_{\max} = 60$.

**Utilidad semántica:**

$$u: V \to [1, 10] \qquad u(v) \in \mathbb{R}$$

Representa la relevancia y pertinencia del curso $v$ respecto al **perfil y objetivos declarados del aprendiz**, tal como son interpretados por el componente LLM del sistema híbrido. Esta función no se define a priori, sino que es **inferida dinámicamente** por el modelo de lenguaje a partir de la descripción en lenguaje natural del curso y del contexto de la consulta del usuario. Formalmente:

$$u(v) = \mathcal{F}_{\text{LLM}}\bigl(\text{desc}(v),\; \text{perfil del aprendiz},\; \text{objetivo de aprendizaje}\bigr)$$

donde $\mathcal{F}_{\text{LLM}}$ denota la función de puntuación semántica implementada por el LLM.

---

### 1.3 Definición del Subgrafo de Ruta Válida

Una **ruta de aprendizaje** se define como un subconjunto de nodos $S \subseteq V$ que satisface la **clausura de prerrequisitos**. Formalmente:

$$\forall\, v_j \in S, \quad \forall\, (v_i, v_j) \in E \;\Rightarrow\; v_i \in S$$

Esto garantiza que si un curso $v_j$ es incluido en la ruta, todos sus prerrequisitos también lo están. El subgrafo inducido $G[S]$ es, por construcción, un DAG.

---

### 1.4 Restricción Dura de Tiempo

El sistema opera bajo una **restricción dura** de tiempo total disponible $T_{\max} \in \mathbb{R}^+$, que acota la suma de las duraciones de todos los cursos seleccionados en la ruta:

$$\sum_{v \in S} d(v) \;\leq\; T_{\max}$$

Esta restricción es **inviolable**: ninguna solución que la incumpla es considerada factible, independientemente de su utilidad total.

---

### 1.5 Función Objetivo

El problema de optimización consiste en encontrar el subconjunto $S^* \subseteq V$ que **maximiza la utilidad semántica acumulada** de los cursos seleccionados, sujeto a la restricción de tiempo y a la clausura de prerrequisitos:

$$S^* = \underset{S \subseteq V}{\arg\max} \sum_{v \in S} u(v)$$

sujeto a:

$$\sum_{v \in S} d(v) \leq T_{\max}$$

$$\forall\, v_j \in S,\; \forall\, (v_i, v_j) \in E \;\Rightarrow\; v_i \in S$$

**Formulación completa como programa de optimización combinatoria:**

Sea $x_v \in \{0, 1\}$ la variable de decisión binaria tal que $x_v = 1$ si el curso $v$ es incluido en la ruta y $x_v = 0$ en caso contrario. El problema se expresa como:

$$\max_{x \in \{0,1\}^{|V|}} \sum_{v \in V} u(v) \cdot x_v$$

sujeto a:

$$\sum_{v \in V} d(v) \cdot x_v \leq T_{\max}$$

$$x_{v_j} \leq x_{v_i} \qquad \forall\, (v_i, v_j) \in E$$

$$x_v \in \{0, 1\} \qquad \forall\, v \in V$$

La segunda familia de restricciones codifica la clausura de prerrequisitos: no es posible seleccionar $v_j$ ($x_{v_j} = 1$) si su prerrequisito $v_i$ no ha sido seleccionado ($x_{v_i} = 0$).

> **Nota de complejidad:** Este problema es una generalización del problema de la mochila con grafos de dependencias (Knapsack with precedence constraints), el cual es **NP-hard** en general. El sistema híbrido propuesto aborda esta complejidad combinando un algoritmo clásico de optimización (e.g., programación dinámica, búsqueda con poda, algoritmo genético) con el LLM como oráculo de utilidad semántica.

---

## 2. Dataset Base — `cursos.json`

El dataset contiene **35 cursos sintéticos** del dominio *"Ciencia de Datos e Inteligencia Artificial"*, organizados en **7 niveles de profundidad** (L0–L6) que forman un DAG verificado. La tabla siguiente resume la estructura:

| Nivel | IDs | Cursos (cantidad) | Prerrequisitos de nivel |
|-------|-----|-------------------|-------------------------|
| L0 | CS_101 – CS_105 | 5 | Ninguno (nodos raíz) |
| L1 | CS_201 – CS_205 | 5 | L0 |
| L2 | CS_301 – CS_305 | 5 | L1 |
| L3 | CS_401 – CS_405 | 5 | L2 |
| L4 | CS_501 – CS_505 | 5 | L3 |
| L5 | CS_601 – CS_605 | 5 | L4 |
| L6 | CS_701 – CS_705 | 5 | L5 |

El archivo `cursos.json` completo se provee como adjunto. Cada objeto sigue el esquema:

```json
{
  "id":             "<String único, ej: 'CS_302'>",
  "titulo":         "<String con nombre del curso>",
  "descripcion":    "<String de 3-5 líneas con temas abstractos y prácticos para análisis semántico por LLM>",
  "duracion_horas": "<Entero en [10, 60]>",
  "prerrequisitos": ["<id_1>", "<id_2>"]
}
```

**Verificación de aciclicidad del grafo:** El orden topológico válido del DAG es:
$$\text{L0} \to \text{L1} \to \text{L2} \to \text{L3} \to \text{L4} \to \text{L5} \to \text{L6}$$

No existe ninguna arista que apunte de un nivel superior a uno inferior, garantizando la propiedad de DAG.

---

## 3. Instancias de Prueba para el Diseño Experimental

Las tres instancias están diseñadas para evaluar el sistema híbrido en condiciones de complejidad creciente, siguiendo el principio de **estrés gradual del optimizador**.

---

### Instancia A — Escenario Pequeño / Simple

**Objetivo:** Validar la corrección del algoritmo en un espacio de búsqueda pequeño con solución óptima verificable manualmente.

**Configuración:**

| Parámetro | Valor |
|-----------|-------|
| Subconjunto de cursos | 10 nodos |
| Cursos incluidos | CS_101, CS_102, CS_201, CS_202, CS_205, CS_302, CS_304, CS_402, CS_501, CS_601 |
| $T_{\max}$ | 80 horas |
| Estructura del grafo | Cadena lineal: CS_101 → CS_201 → CS_302 → CS_402 → CS_501 → CS_601, con rama paralela CS_102 → CS_202 → CS_304 |

**Grafo de dependencias (Instancia A):**

```
CS_101 ──► CS_201 ──► CS_302 ──► CS_402 ──► CS_501 ──► CS_601
CS_102 ──► CS_202 ──► CS_304 ──────────────────────────────┘
CS_205 (depende de CS_101)
```

**Duraciones relevantes:**

| ID | Duración (h) |
|----|-------------|
| CS_101 | 20 |
| CS_102 | 25 |
| CS_201 | 25 |
| CS_202 | 30 |
| CS_205 | 30 |
| CS_302 | 40 |
| CS_304 | 30 |
| CS_402 | 50 |
| CS_501 | 45 |
| CS_601 | 50 |

**Comportamiento esperado:**

Con $T_{\max} = 80$ h, el espacio de rutas válidas es muy reducido. El algoritmo debe identificar rápidamente (en tiempo despreciable) que ninguna ruta que incluya CS_402 (50 h) junto con sus prerrequisitos es factible dentro del límite. La solución óptima esperada es la ruta más corta que maximiza la utilidad: probable combinación de cursos fundacionales de bajo costo (CS_101 + CS_201 + CS_102 + CS_202, suma = 100 h → inviable completa), lo que obliga al optimizador a explorar rutas parciales y a demostrar que gestiona correctamente la clausura de prerrequisitos. Este escenario sirve como **prueba de sanidad** (*sanity check*) del sistema.

---

### Instancia B — Escenario Mediano / Intermedio

**Objetivo:** Evaluar el comportamiento del algoritmo ante ramificaciones en el grafo y múltiples trayectorias competidoras hacia un mismo nodo destino.

**Configuración:**

| Parámetro | Valor |
|-----------|-------|
| Subconjunto de cursos | 22 nodos |
| Cursos incluidos | CS_101 a CS_105 (L0) + CS_201 a CS_205 (L1) + CS_301, CS_302, CS_303, CS_304, CS_305 (L2) + CS_401, CS_402, CS_403, CS_404, CS_405 (L3) + CS_503 (L4) |
| $T_{\max}$ | 200 horas |
| Estructura del grafo | Árbol con ramificaciones: múltiples nodos de L2 convergiendo en L3; CS_401 y CS_404 son prerrequisitos conjuntos de CS_503 |

**Ramificaciones clave:**

- CS_302 requiere **tres** prerrequisitos: {CS_202, CS_203, CS_201} → alta barrera de entrada.
- CS_402 y CS_401 compiten por el presupuesto temporal al requerir solapamiento de prerrequisitos.
- CS_503 requiere {CS_401, CS_404} → exige la convergencia de dos ramas del grafo.

**Comportamiento esperado:**

El algoritmo enfrenta un **tradeoff real**: la ruta hacia CS_503 (sistemas de recomendación) implica construir toda la base de L0 a L3, lo cual puede exceder $T_{\max}$ si se acumulan todos los prerrequisitos. El LLM debe discriminar semánticamente qué caminos hacia L3 son más relevantes para el perfil del usuario. Se espera que el optimizador exhiba **poda efectiva** de ramas sub-óptimas y que el LLM muestre su valor añadido al asignar mayores puntuaciones $u(v)$ a nodos alineados con el perfil declarado. Este escenario permite medir la **ganancia de utilidad** del sistema híbrido frente a un baseline de recorrido topológico sin puntuación semántica.

---

### Instancia C — Escenario Grande / Complejo

**Objetivo:** Estresar el optimizador con el espacio de búsqueda completo y un presupuesto restrictivo que impide incluir más del 50% de los cursos, forzando decisiones de alta discriminación semántica.

**Configuración:**

| Parámetro | Valor |
|-----------|-------|
| Conjunto de cursos | **35 nodos** (dataset completo) |
| $T_{\max}$ | 300 horas |
| Estructura del grafo | DAG denso de 7 niveles; nodos L5 y L6 tienen dependencias cruzadas entre ramas de NLP, Visión y RL |
| Duración total del dataset | $\sum_{v \in V} d(v) = 1295$ horas |
| Fracción máxima seleccionable | $\approx 23\%$ del tiempo total disponible |

**Complejidad estructural:**

- CS_703 (Agentes Autónomos) requiere {CS_601, CS_605} → convergencia de la rama NLP con la rama de Deep RL, dos trayectorias largas e independientes.
- CS_704 (IA Generativa) requiere {CS_602, CS_601} → convergencia de la rama de Visión con la de NLP.
- CS_705 (Proyecto Integrador) requiere {CS_701, CS_703, CS_604} → nodo sumidero de alta exigencia que presupone casi toda la cadena del grafo.
- Varias rutas hacia nodos de L5/L6 comparten prerrequisitos, creando **interdependencias no triviales** entre decisiones de selección.

**Comportamiento esperado:**

Con $T_{\max} = 300$ h, el número de rutas factibles que satisfacen la clausura de prerrequisitos es exponencial en teoría pero drásticamente reducido por la restricción temporal. El optimizador debe demostrar **escalabilidad** y **eficiencia de poda**. El LLM se vuelve crítico como guía heurística: sin la señal semántica de $u(v)$, un algoritmo de búsqueda exhaustiva tomaría tiempo prohibitivo o convergería a soluciones de baja calidad. Este escenario permite cuantificar:

1. El **tiempo de cómputo** del optimizador en función del tamaño del grafo.
2. La **mejora de calidad** (utilidad total) aportada por las puntuaciones del LLM frente a utilidades uniformes ($u(v) = 5$ para todo $v$).
3. La **consistencia semántica** de la ruta generada según el perfil del aprendiz.

---

## 4. Resumen Comparativo de las Instancias

| Parámetro | Instancia A | Instancia B | Instancia C |
|-----------|-------------|-------------|-------------|
| $|V|$ (nodos) | 10 | 22 | 35 |
| $|E|$ (aristas) | ~8 | ~22 | ~42 |
| $T_{\max}$ (horas) | 80 | 200 | 300 |
| Profundidad máxima del DAG | 6 | 4 | 7 |
| Ramificaciones | Lineal | Moderada | Alta / cruzada |
| Rol principal del LLM | Validación semántica básica | Discriminación entre rutas competidoras | Guía heurística crítica para la poda |
| Comportamiento esperado | Solución verificable manualmente | Tradeoff ruta-costo observable | Estrés del optimizador; LLM determinante |
