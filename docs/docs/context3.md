# Objetivos
- Integrar al repositorio del proyecto integrador un conjunto de pruebas automatizadas (con PyTest y DeepChecks) y chequeos de seguridad básicos, con el fin de asegurar la calidad, confiabilidad y buenas prácticas de seguridad del código y los datos del proyecto.
- Verificar mediante pruebas automatizadas que el entrenamiento de modelos sea reproducible (por ejemplo, con semillas fijas y resultados consistentes).

Esta actividad corresponde a la fase colaborativa del entregable de la semana y refuerza las competencias de testing y gobernanza técnica sobre el proyecto que el equipo ha venido construyendo.

# Instrucciones
En esta actividad colaborativa, tu equipo añadirá una capa de calidad y seguridad al repositorio del proyecto. El objetivo es que el código cuente con pruebas que se puedan ejecutar de forma automatizada y que se hayan revisado los riesgos de seguridad más comunes.

Realiza, junto con tu equipo, los siguientes pasos:

1. Escribe pruebas unitarias con PyTest: Crea pruebas para al menos tres funciones clave del proyecto (procesamiento de datos, feature engineering, entrenamiento o utilidades), cubriendo tanto casos normales como casos límite. Incluye al menos una prueba que verifique la reproducibilidad del entrenamiento (por ejemplo, que dos ejecuciones con la misma semilla produzcan las mismas predicciones o métricas).
2. Integra validación de datos con DeepChecks: Añade chequeos que validen la integridad y la calidad del dataset (por ejemplo, tipos de datos, valores faltantes, duplicados o drift entre conjuntos de entrenamiento y prueba).
3. Realiza chequeos de seguridad básicos: Revisa que el repositorio no exponga credenciales, contraseñas ni datos sensibles (usa un archivo .gitignore y variables de entorno), y ejecuta al menos una herramienta de análisis de seguridad de dependencias o de código (por ejemplo, pip-audit o bandit).
4. Organiza las pruebas en el repositorio: Ubica las pruebas en una carpeta dedicada (por ejemplo, tests/) siguiendo la estructura del proyecto, de modo que se puedan ejecutar con un solo comando.
5. Documenta la ejecución: Actualiza el README con las instrucciones para correr las pruebas y los chequeos, e incluye evidencia (capturas) de su ejecución exitosa.
6. Refleja el trabajo colaborativo: Asegura que las contribuciones de todos los integrantes queden registradas en el historial de Git mediante ramas y commits significativos.