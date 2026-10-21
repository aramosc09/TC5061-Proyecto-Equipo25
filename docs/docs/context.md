# Proyecto MiBici: propuesta y base reproducible
## Objetivo
propondrán cómo resolver con Machine Learning un conjunto de preguntas sobre la operación de MiBici, el sistema de bicicleta pública del Área Metropolitana de Guadalajara, y dejarán montado el repositorio sobre el que construirán el proyecto.

### Sobre los datos
- Viajes por mes en archivos CSV (`datos_abiertos_AAAA_MM.csv`) desde diciembre de 2014. Cada registro es un viaje con identificador, datos del usuario, fecha y hora de inicio y fin, y estaciones de origen y destino.
- Nomenclatura de estaciones (`nomenclatura_2026_09.csv`) con identificador, nombre, latitud, longitud y capacidad de cada estación.
Los datos ya estan descargados y disponibles aquí: data/raw

## Preguntas que debe responder el proyecto
Tres preguntas. La forma de plantearlas como problema de ML (variable objetivo, granularidad, horizonte, modelos y métricas) es decisión del equipo y debe justificarse.

| # | Pregunta de negocio | Tipo de tarea| Quien la usaría |
| --- | --- | --- | --- |
| P1 | ¿Cuántos viajes se iniciarán en una estación o zona en un periodo futuro? | Regresión| Operador del sistema, para planear la flota. |
| P2 | ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes disponibles en un periodo futuro? | Clasificación | Equipo de rebalanceo de bicicletas. |
| P3 | ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican en la ciudad? | Clustering| Planeación urbana y expansión del sistema. |

Los datos de MiBici no incluyen el inventario de bicicletas por estación. Para P2 deberán definir y justificar una forma de aproximar el riesgo a partir de los viajes y la capacidad de cada estación.

La actividad se divide en dos partes complementarias.

## Parte A. Propuesta de solución

1. Formular el problema: Llenen un ML Canvas para el proyecto y, para cada pregunta, describan: 1) Qué se predice o se agrupa, con qué granularidad y horizonte, 2) Por qué es relevante para la operación de MiBici y 3) Quién usaría el resultado y qué decisión tomaría con él.
2. Definir los datos: Propongan y justifiquen el rango de fechas que usarán y documenten: 1) Fuente (enlaces a los archivos), 2) Tamaño real obtenido (meses, número de viajes y número de estaciones), 3) Variables originales que usarán y cómo construyen cada variable objetivo, 4) Condiciones de uso publicadas por MiBici y el tratamiento que darán a los datos personales, y 5) La estrategia de partición de los datos, explicando cómo evitan usar información del futuro.
3. Diseñar el pipeline de ML: Describan, para cada pregunta, las etapas desde los datos crudos hasta el modelo evaluado: limpieza, agregaciones, feature engineering, entrenamiento y evaluación. Indiquen qué variables construirán y por qué son pertinentes.
4. Proponer modelos y métricas: Para cada pregunta, definan un baseline sencillo, al menos un modelo candidato y las métricas con las que lo evaluarán. Justifiquen por qué esas métricas son coherentes con el tipo de tarea y con la decisión que se tomará (consulten la guía de evaluación de modelos de scikit-learn).
5. Proponer la arquitectura del sistema: Presenten un diagrama que muestre los componentes de su solución y cómo se conectan: fuentes de datos, ingesta, almacenamiento, procesamiento, entrenamiento y registro de experimentos. Declaren las herramientas que usarán en cada componente y justifiquen su elección de acuerdo a las lecturas de la semana.
6. Ejecutar el experimento base en MLFlow: Implementen los baselines y regístrenlos en MLFlow Tracking (ver la guía rápida y la API de tracking):
- Un experimento cuyo nombre incluya el número de equipo (por ejemplo, mibici-equipo03).
- Al menos un run por cada pregunta (P1, P2 y P3), con nombre descriptivo.
- En cada run: los parámetros que permitan reproducirlo (incluyendo el rango de fechas y la versión de los datos), las métricas que propusieron y al menos un artefacto (por ejemplo, una gráfica o un mapa de los grupos de estaciones).
- Etiquetas que identifiquen al equipo, a la pregunta (P1, P2 o P3) y el commit de Git con que se ejecutó.
- Los runs se ejecutan desde el código del repositorio, no solo desde un notebook.
7. Capturar la evidencia: Tomen capturas de pantalla de la interfaz de MLFlow que muestren:
- La lista de runs del experimento.
- El detalle de un run con sus parámetros, métricas y etiquetas.
- Al menos un artefacto registrado.
- La comparación de dos o más runs.

## Parte B

8. Generar la estructura base: Un integrante instala la plantilla con pipx (pipx install cookiecutter-data-science), ejecuta ccds con la opción include_code_scaffold en Yes y sube el resultado como primer commit. El resto de las opciones las decide el equipo. Sobre esa estructura:
- Respeten la separación de carpetas de la plantilla: datos en data/, notebooks en notebooks/, código en el módulo del proyecto, modelos en models/ y figuras en reports/figures/.
- Centralicen rutas, fechas y semillas en el archivo config.py del módulo.
- Fijen las versiones de las dependencias en el archivo que eligieron al generar la plantilla.
- Amplíen el .gitignore para excluir los datos de MiBici, las carpetas que genera MLFlow (mlruns/, mlartifacts/), archivos .env y modelos serializados.
- Guarden el ML Canvas y el diagrama de arquitectura en references/.
9. Establecer el flujo de trabajo con Git/GitHub: Definan y documenten en el README su flujo, que debe incluir:
- Rama principal protegida; todo cambio entra mediante Pull Request siguiendo GitHub Flow.
- Una convención de nombres de ramas y de commits (por ejemplo, Conventional Commits).
- Cada integrante abre al menos un Pull Request y hace al menos una revisión a otro integrante.
10. Construir la ingesta reproducible: En el archivo dataset.py del módulo, implementen un proceso ejecutable desde la terminal que, sin pasos manuales:
- Descargue los meses que eligieron y la nomenclatura de estaciones.
- Normalice nombres de columnas, codificación de caracteres y tipos de fecha.
- Guarde los datos limpios en un formato eficiente para grandes volúmenes (por ejemplo, Parquet).
- Versione los datos con el mecanismo que propongan. Una opción sencilla es un manifiesto versionado en Git con el número de registros y el hash (hashlib) de cada archivo. La versión de los datos debe poder registrarse en MLFlow.
11. Desarrollar el módulo de feature engineering: Trasladen las transformaciones de su notebook de exploración a funciones reutilizables en el archivo features.py del módulo, con docstring que explique qué hace cada una y por qué es pertinente para la pregunta que atiende. Las variables no deben usar información posterior al momento de la predicción.
12. Redactar el README: Documenten el propósito del proyecto, las tres preguntas, los datos utilizados, la estructura de carpetas, la instalación, los comandos de ingesta y del experimento base, el flujo de trabajo con Git y los integrantes del equipo con sus roles.

Nota: Herramientas opcionales. El uso de herramientas de linting y formateo (por ejemplo, las que ofrece la plantilla al generarse) es opcional y no se evalúa.

## Buenas Prácticas Obligatorias

### MLFlow
1. Un experimento por proyecto y runs con nombre descriptivo, nunca el nombre aleatorio por defecto. Se verifica en: Interfaz de MLFlow
2. Registrar todo lo necesario para reproducir el run: rango de fechas, versión de los datos, semilla y modelo con sus hiperparámetros. Se verifica en: Parámetros del run.
3. Ligar cada run con el código que lo produjo (commit de Git) y con la pregunta que atiende. Se verifica en: Etiquetas del run
4. Nombrar las métricas indicando el conjunto de datos en que se calculan (por ejemplo, test_mae en lugar de mae). Se verifica en: Métricas del run
5 . No registrar credenciales ni datos personales como parámetros, etiquetas o artefactos. Se verifica en: Interfaz de MLFlow y código

### Estructura y código
1. Notebooks solo para exploración, nombrados con la convención de la plantilla (1.0-iniciales-descripcion.ipynb); la lógica final vive en el módulo. Se verifica en: Carpeta notebooks/
2. Configuración en un solo lugar: sin rutas, fechas ni semillas repetidas en varios archivos. Se verifica en: config.py
3. Dependencias con versión fija y entorno reproducible desde un clon limpio. Se verifica en: Archivo de dependencias y README

### Datos y git
1. Ningún dato de MiBici, modelo, carpeta de MLFlow ni credencial en el historial de Git. Se verifica en: .gitignore e historial
2. Todo cambio entra a la rama principal por Pull Request revisado. Se verifica en: Pestaña de Pull Requests

## Especificaciones de entrega
| Ámbito | Prácticas |
|---|---|
| PDF con la propuesta | Enlace al repositorio; ML Canvas y formulación de P1, P2 y P3; definición de los datos y partición; pipeline de ML; baselines, modelos candidatos y métricas por pregunta; diagrama de arquitectura con herramientas justificadas; resultados del experimento base; capturas de MLFlow (paso 7); captura de la estructura del repositorio; referencias en formato APA 7. |
| Repositorio de GitHub | Estructura generada con `ccds`; ingesta en `dataset.py` con su mecanismo de versionado; `features.py` documentado; código del experimento base; al menos un notebook de exploración; ML Canvas y diagrama en `references/`; README completo; `.gitignore` ampliado; dependencias con versión fija. |
| Evidencia de colaboración | Rama principal protegida; historial con commits de los cuatro integrantes; al menos un Pull Request abierto y una revisión por integrante. |
| Experimento base en MLFlow | Al menos un run por pregunta con parámetros, métricas, etiquetas y artefactos, evidenciado con las capturas del paso 7 y reproducible con el comando documentado en el README. |