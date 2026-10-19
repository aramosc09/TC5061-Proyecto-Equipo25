# Validación y calidad de datos del proyecto MiBici

**Materia:** TC5061 - Operaciones de Aprendizaje Automático (MLOps)
**Proyecto:** Análisis y predicción del sistema de bicicletas públicas MiBici
**Equipo:** 25
**Periodo analizado:** Octubre 2023 - Septiembre 2026

---

## 1. Objetivo del documento

El presente documento describe las actividades de revisión,
validación y análisis de calidad realizadas sobre los datos
históricos del sistema de bicicletas públicas MiBici.

El propósito es identificar inconsistencias, verificar la
integridad de los registros y establecer criterios para
preparar los datos que utilizarán los modelos de
Machine Learning del proyecto.

Las verificaciones realizadas incluyen:

- Revisión de la estructura de los archivos.
- Identificación del periodo histórico disponible.
- Selección del periodo de análisis.
- Conteo de registros por mes.
- Revisión de valores faltantes.
- Identificación de estaciones históricas.
- Comparación con la nomenclatura oficial.
- Validación de capacidad y coordenadas.
- Análisis de estaciones sin correspondencia.
- Recomendaciones para P1, P2 y P3.

Los resultados documentados corresponden a verificaciones
exploratorias realizadas con Python y PyArrow.

---

## 2. Fuentes de información

Los datos utilizados provienen del portal oficial de
datos abiertos de MiBici.

Fuente:

https://www.mibici.net/es/datos-abiertos/

Se utilizaron dos conjuntos principales:

### 2.1. Registros históricos de viajes

Los registros contienen información sobre los viajes
realizados en el sistema de bicicletas públicas.

Entre las variables disponibles se encuentran:

- Viaje_Id
- Usuario_Id
- Genero
- Año_de_nacimiento
- Inicio_del_viaje
- Fin_del_viaje
- Origen_Id
- Destino_Id

Durante la consolidación se incorpora una columna
que identifica el mes de procedencia de cada registro.

### 2.2. Nomenclatura de estaciones

Se utilizó el archivo:

`nomenclatura_2026_09.csv`

El archivo incluye las siguientes variables:

- id
- name
- latitude
- longitude
- dpcapacity

Estas variables permiten identificar las estaciones,
obtener su ubicación geográfica y consultar su capacidad.

---

## 3. Revisión del notebook exploratorio

Como punto de partida se revisó el notebook de
análisis exploratorio desarrollado por el equipo.

En este trabajo se identificaron:

- 142 archivos mensuales de viajes.
- Información histórica desde diciembre de 2014.
- Archivos con diferentes codificaciones.
- Procedimientos de consolidación de registros.
- Análisis de cobertura y distribución temporal.

### 3.1. Codificaciones identificadas

Durante la revisión se encontraron archivos con
diferentes codificaciones de caracteres:

| Codificación | Archivos |
|--------------|----------|
| UTF-8 | 99 |
| CP1252 | 43 |
| Total | 142 |

Esto es relevante porque la lectura de todos
los archivos utilizando una única codificación
podría provocar errores de importación.

Se recomienda que el módulo de ingesta contemple
las codificaciones identificadas.

### 3.2. Observaciones sobre el procesamiento

Durante la revisión del notebook se identificaron
oportunidades de mejora:

1. Evitar depender de rutas locales específicas.
2. Normalizar los nombres de las columnas.
3. Validar explícitamente los formatos de fechas.
4. Separar exploración y procesamiento reproducible.
5. Registrar los criterios de limpieza utilizados.
6. Centralizar configuraciones del proyecto.
7. Generar un manifiesto de los datos procesados.

Estas recomendaciones permiten trasladar el
procesamiento exploratorio a un pipeline reproducible.

---

## 4. Periodo seleccionado

Se decidió analizar el periodo comprendido entre:

**Octubre de 2023 y septiembre de 2026**

Este intervalo corresponde a:

- 36 meses consecutivos.
- Tres ciclos anuales de actividad.
- Información relativamente reciente.
- Un periodo posterior a las principales
  restricciones de movilidad asociadas con COVID-19.

El uso de tres años permite analizar patrones
temporales y contar con suficiente información
para entrenamiento, validación y prueba.

---

## 5. Estructura del archivo consolidado

Se revisó el archivo:

`raw_data.parquet`

### 5.1. Información general

| Característica | Resultado |
|----------------|-----------|
| Formato | Parquet |
| Registros históricos totales | 39,078,984 |
| Columnas | 9 |
| Grupos de filas | 38 |
| Tamaño aproximado | 737 MB |

### 5.2. Columnas identificadas

Las columnas del archivo consolidado son:

1. Viaje_Id
2. Usuario_Id
3. Genero
4. Año_de_nacimiento
5. Inicio_del_viaje
6. Fin_del_viaje
7. Origen_Id
8. Destino_Id
9. mes_archivo

Durante la revisión se observó que las columnas
se encontraban almacenadas principalmente
como cadenas de texto.

Por ello, antes del modelado es necesario
realizar conversiones explícitas de tipos.

### 5.3. Formato de fechas y periodo

La columna `mes_archivo` utiliza el formato:

`AAAA-MM`

Las fechas de inicio y fin se presentan
en un formato como:

`AAAA-MM-DD HH:MM:SS`

Estos formatos permiten construir
variables temporales para los modelos.

---

## 6. Selección del periodo de análisis

Se filtraron los registros utilizando
la columna `mes_archivo`.

Condiciones aplicadas:

- mes_archivo >= 2023-10
- mes_archivo <= 2026-09

### Resultado

**Registros del periodo: 13,621,885 viajes**

**Meses identificados: 36**

La cantidad de viajes coincidió con el
resultado obtenido previamente en el
notebook exploratorio del equipo.

Esta coincidencia permitió corroborar
el volumen del periodo seleccionado.

---

## 7. Distribución mensual de viajes

La siguiente tabla presenta el número
de registros identificados por mes.

| Mes | Viajes |
|-----|--------|
| 2023-10 | 383,239 |
| 2023-11 | 384,464 |
| 2023-12 | 314,164 |
| 2024-01 | 391,591 |
| 2024-02 | 383,732 |
| 2024-03 | 390,513 |
| 2024-04 | 405,308 |
| 2024-05 | 410,806 |
| 2024-06 | 365,000 |
| 2024-07 | 364,177 |
| 2024-08 | 394,894 |
| 2024-09 | 390,072 |
| 2024-10 | 418,690 |
| 2024-11 | 391,089 |
| 2024-12 | 340,049 |
| 2025-01 | 380,184 |
| 2025-02 | 372,529 |
| 2025-03 | 405,110 |
| 2025-04 | 366,193 |
| 2025-05 | 395,891 |
| 2025-06 | 356,611 |
| 2025-07 | 368,186 |
| 2025-08 | 374,727 |
| 2025-09 | 376,789 |
| 2025-10 | 420,064 |
| 2025-11 | 380,328 |
| 2025-12 | 335,420 |
| 2026-01 | 357,180 |
| 2026-02 | 333,756 |
| 2026-03 | 376,243 |
| 2026-04 | 351,851 |
| 2026-05 | 315,895 |
| 2026-06 | 355,303 |
| 2026-07 | 388,257 |
| 2026-08 | 393,524 |
| 2026-09 | 490,056 |

### Observaciones

El periodo presenta variaciones mensuales
en el número de viajes registrados.

Estas variaciones pueden relacionarse
con patrones estacionales y cambios
en el uso del sistema.

Sin embargo, para identificar sus causas
sería necesario integrar información adicional.

El incremento observado en septiembre de 2026
debe analizarse con mayor detalle antes
de atribuirlo a alguna causa particular.

---

## 8. Validación básica de integridad

Se realizó una revisión de valores faltantes
en las siguientes columnas:

- Origen_Id
- Destino_Id
- Inicio_del_viaje
- Fin_del_viaje

### Resultado

Se identificaron:

**0 valores faltantes o vacíos en estas cuatro columnas.**

Este resultado corresponde a la revisión
de integridad básica realizada sobre los
13,621,885 registros seleccionados.

La ausencia de valores faltantes no garantiza
por sí sola que todos los registros sean válidos.

También es necesario considerar:

- Formatos correctos de fechas.
- Duraciones negativas o anómalas.
- Duplicados.
- Inconsistencias temporales.
- Cobertura histórica de las estaciones.

Estas verificaciones deben formar parte
del pipeline definitivo.

---

## 9. Identificación de estaciones históricas

Se analizaron los identificadores
de las estaciones de origen y destino.

### Resultados

| Indicador | Resultado |
|-----------|-----------|
| IDs únicos de origen | 473 |
| IDs únicos de destino | 473 |
| IDs únicos considerando ambos | 473 |

Por lo tanto, durante el periodo seleccionado
se identificaron:

**473 estaciones diferentes en los viajes.**

Este resultado permite comparar la cobertura
histórica con la nomenclatura disponible.

---

## 10. Comparación con la nomenclatura

Se utilizó el archivo:

`nomenclatura_2026_09.csv`

Se normalizaron los identificadores
de las estaciones para realizar
la comparación con los viajes.

### Resultados

| Indicador | Resultado |
|-----------|-----------|
| Registros del catálogo | 484 |
| IDs únicos del catálogo | 484 |
| IDs únicos en los viajes | 473 |
| IDs coincidentes | 471 |
| IDs de viajes sin catálogo | 2 |
| IDs del catálogo sin viajes | 13 |

### Interpretación

De las 473 estaciones identificadas
en los viajes, 471 tienen correspondencia
con la nomenclatura disponible.

Esto representa aproximadamente el 99.58%
de los identificadores históricos.

Es importante distinguir esta proporción
de la cantidad de viajes afectados.

La coincidencia por identificador
no representa necesariamente la
misma proporción de registros.

---

## 11. Estaciones sin correspondencia

Se identificaron dos estaciones
presentes en los viajes que no aparecen
en la nomenclatura de septiembre de 2026.

Los identificadores son:

- 405
- 407

Estas estaciones fueron analizadas
para conocer su actividad histórica.

### 11.1. Estación 405

Resultados:

- Salidas: 3
- Llegadas: 3
- Meses con actividad: 1
- Primer mes: 2026-06
- Último mes: 2026-06

La actividad registrada para esta estación
es muy reducida y se concentra en junio de 2026.

### 11.2. Estación 407

Resultados:

- Salidas: 1,957
- Llegadas: 2,120
- Meses con actividad: 2
- Primer mes: 2026-06
- Último mes: 2026-07

Esta estación presenta una actividad
considerablemente mayor que la estación 405.

Sin embargo, tampoco tiene correspondencia
en la nomenclatura de septiembre de 2026.

### 11.3. Impacto general

Se identificaron:

**3,954 viajes con origen o destino
en las estaciones 405 o 407.**

Esto representa:

**0.0290% del total de viajes del periodo.**

El conteo corresponde a viajes únicos
afectados por alguna de estas estaciones.

No se obtiene sumando directamente
todas las salidas y llegadas, porque
un viaje puede involucrar las dos estaciones
o generar conteos en ambas direcciones.

### Interpretación

El impacto sobre el conjunto completo
de registros es pequeño.

Sin embargo, las estaciones 405 y 407
no deberían utilizarse en cálculos que
requieren capacidad o coordenadas
hasta contar con metadatos confiables.

---

## 12. Estaciones del catálogo sin viajes

Se encontraron 13 identificadores
presentes en la nomenclatura que no
aparecen en los viajes del periodo.

Los identificadores son:

- 22
- 97
- 99
- 109
- 113
- 114
- 116
- 138
- 147
- 149
- 223
- 250
- 492

La ausencia de viajes en el periodo
no demuestra por sí sola que estas
estaciones estuvieran fuera de servicio.

Para conocer su situación sería necesario
consultar información histórica adicional.

---

## 13. Validación del catálogo

Se revisaron las siguientes variables:

- Identificador de estación.
- Capacidad.
- Latitud.
- Longitud.

### Resultados

| Indicador | Resultado |
|-----------|-----------|
| IDs duplicados | 0 |
| Capacidades faltantes | 0 |
| Capacidades menores o iguales a cero | 4 |
| Latitudes faltantes | 0 |
| Longitudes faltantes | 0 |
| Estaciones coincidentes con metadatos válidos | 471 |

### 13.1. Estaciones con capacidad cero

Se identificaron cuatro estaciones
con capacidad igual a cero.

| ID | Estación | Capacidad |
|----|----------|-----------|
| 99 | ex-(ZPN-009) C. Del Corral / C. Nicolás Bravo | 0 |
| 109 | ex-(ZPN-019) C. Cabecera Mpal. Juan Manuel / C. | 0 |
| 116 | ex-(ZPN-026) C. 16 Sep. / C. Fray A. de Segovia | 0 |
| 138 | ex-(ZPN-044) | 0 |

Estas cuatro estaciones no aparecen
entre las 471 estaciones coincidentes
con los viajes del periodo.

El prefijo `ex-` podría indicar que
corresponden a estaciones antiguas,
pero esto requiere confirmación oficial.

### 13.2. Relevancia para P2

La capacidad es necesaria para
calcular el balance normalizado
utilizado como etiqueta proxy de P2.

Por lo tanto, es importante evitar
divisiones entre cero y utilizar
únicamente capacidades válidas.

---

## 14. Recomendaciones de tratamiento de datos

Las siguientes decisiones se proponen
a partir de los resultados obtenidos.

### 14.1. Conservar los archivos originales

Se recomienda conservar localmente
una copia sin modificar de los datos
originales para asegurar trazabilidad.

Los archivos originales no deben
subirse al repositorio de GitHub.

### 14.2. No eliminar viajes innecesariamente

Si un viaje involucra una estación
sin correspondencia con el catálogo,
no necesariamente debe descartarse
el registro completo.

Por ejemplo:

Un viaje que comienza en la estación 47
y termina en la estación 407 permite
contabilizar correctamente una salida
desde la estación 47.

Sin embargo, la estación 407 no
podrá utilizarse en cálculos que
necesiten su capacidad o ubicación
mientras no se disponga de esos datos.

### 14.3. Conservar trazabilidad de las exclusiones

Se recomienda documentar:

- Número de registros originales.
- Registros descartados.
- Motivo de cada exclusión.
- Estaciones afectadas.
- Periodos afectados.
- Versión de los datos procesados.

Esta información debe integrarse
en el mecanismo de versionado del proyecto.

---

## 15. Aplicación de resultados a los modelos

### 15.1. P1 - Regresión

P1 tiene como objetivo predecir
el número de viajes iniciados
en una estación durante una hora futura.

Se recomienda conservar el historial
de las estaciones con registros suficientes.

La ausencia de metadatos no impide
necesariamente modelar los conteos
históricos de una estación.

La selección final dependerá
de la cantidad y calidad de
información disponible.

También deben considerarse las
horas sin viajes de estaciones
que se sabe que estaban activas.

No se deben crear automáticamente
horas con cero viajes para periodos
en los que la estación no operaba
o no existe evidencia de cobertura.

### 15.2. P2 - Clasificación

P2 utiliza un indicador aproximado
de presión basado en el balance
de llegadas y salidas.

La formulación propuesta utiliza:

Balance = llegadas - salidas

Balance normalizado = Balance / capacidad

Por lo tanto, se recomienda utilizar
estaciones con capacidad válida.

Inicialmente se dispone de
471 estaciones coincidentes
con información utilizable.

Es importante recordar que
este indicador no demuestra
la existencia de un evento real
de desabasto o saturación.

### 15.3. P3 - Clustering

P3 busca identificar estaciones
con patrones históricos similares.

Se propone utilizar:

- Perfiles de actividad semanal.
- Conteos o proporciones de salidas.
- Conteos o proporciones de llegadas.
- Indicadores de cobertura temporal.

Las coordenadas de las estaciones
se utilizarán para interpretar
y representar los grupos en un mapa.

Se recomienda trabajar inicialmente
con las 471 estaciones coincidentes
que cuentan con metadatos válidos.

También debe evaluarse la cobertura
temporal de cada estación antes de
construir y comparar sus perfiles.

---

## 16. Limitaciones identificadas

### 16.1. Nomenclatura histórica

El archivo utilizado corresponde
a septiembre de 2026.

No se cuenta en esta revisión con
un historial completo de cambios
de capacidad y ubicación.

Por lo tanto, no se puede asegurar
que las capacidades observadas
fueran las mismas durante los 36 meses.

### 16.2. Disponibilidad real de bicicletas

Los registros de viajes permiten
observar entradas y salidas.

Sin embargo, no permiten conocer
directamente el inventario de bicicletas
disponibles en cada momento.

Tampoco incluyen necesariamente
todos los movimientos de rebalanceo
realizados por el operador.

Esto limita la interpretación
de P2 como riesgo operativo real.

### 16.3. Cobertura temporal

Una estación puede aparecer
únicamente durante algunos meses.

Por esta razón, la ausencia
de registros no debe interpretarse
automáticamente como demanda cero.

### 16.4. Calidad de registros

Aunque no se encontraron valores
faltantes en las cuatro columnas
principales revisadas, deben mantenerse
validaciones automáticas sobre:

- Fechas.
- Duraciones.
- Duplicados.
- Identificadores.
- Consistencia temporal.

---

## 17. Próximas actividades recomendadas

Para continuar con la preparación
reproducible de datos se proponen
las siguientes actividades:

1. Confirmar los resultados de limpieza
   mediante el pipeline definitivo.

2. Documentar las reglas de exclusión
   y los registros afectados.

3. Consolidar una interfaz común
   de datos para P1, P2 y P3.

4. Generar agregaciones por
   estación y hora.

5. Identificar periodos de cobertura
   de cada estación.

6. Implementar funciones reutilizables
   de ingeniería de características.

7. Aplicar particiones temporales
   sin fuga de información futura.

8. Registrar versiones de datos
   y parámetros en MLflow.

9. Documentar los comandos necesarios
   para reproducir el proceso.

10. Incorporar pruebas automáticas
    de calidad de datos.

---

## 18. Conclusiones

La revisión permitió confirmar
que el periodo de octubre de 2023
a septiembre de 2026 contiene
13,621,885 registros de viajes.

También se identificaron
473 estaciones históricas,
de las cuales 471 tienen
correspondencia con la
nomenclatura de septiembre de 2026.

Las dos estaciones sin catálogo
representan un impacto reducido
sobre el conjunto de registros,
equivalente al 0.0290% de los viajes.

Adicionalmente, se identificaron
cuatro estaciones con capacidad cero
en la nomenclatura, ninguna de ellas
presente entre las estaciones
coincidentes.

Estos resultados proporcionan una
base para definir criterios de
limpieza, transformación y selección
de estaciones para los modelos
P1, P2 y P3.

La documentación de las decisiones
de calidad y el versionado de los
datos serán fundamentales para
garantizar la reproducibilidad
del proyecto.

---

## 19. Referencia de los datos

MiBici. (s. f.). Datos abiertos.

https://www.mibici.net/es/datos-abiertos/