# Entrega 3 — Ejecución y evidencias de PyTest, Deepchecks y seguridad

Documento de apoyo para reproducir los chequeos de la actividad **Tests + chequeos de
seguridad básicos integrados al repositorio** del Equipo 25. Registrar la fecha,
commit y rama para que las capturas se asocien con el código realmente evaluado.
No declarar resultados que no se hayan comprobado mediante una ejecución real.

## 1. Preparar el entorno

```bash
git branch --show-current
git rev-parse --short HEAD
git status --short
uv sync --frozen
```

- Utilizar Python 3.13 y `uv.lock` para PyTest, Bandit y pip-audit principal.
- Deepchecks 0.19.1 se ejecuta aparte con Python 3.12 y
  `requirements-quality.txt`.
- Antes de subir evidencias, verificar que no aparezcan `data/data_manifest.json`,
  `.env`, bases MLflow, Parquet ni archivos con filas individuales en `git status`.

## 2. Pruebas automatizadas

```bash
uv run --frozen python -m pytest tests -q
# Alternativa: make test
```

El equipo ejecutó la suite sobre la rama temporal `review/pr19` en macOS con el
resultado **80 passed, 18 warnings**. Las advertencias reportadas provinieron de
Matplotlib/Pyparsing y opciones del optimizador de Scikit-learn; no fallaron
pruebas. Este dato corresponde a esa ejecución específica: repetir en el commit
final antes de tomar la captura.

Reproducibilidad: `tests/test_candidate_models.py` contiene
`test_random_forest_training_is_reproducible_with_same_seed`, que compara dos
entrenamientos con la misma semilla; `test_data_quality.py` comprueba además
muestreo reproducible.

**Evidencia solicitada:** captura de la terminal que muestre el comando completo,
la línea `80 passed` (o el número real de la ejecución final) y la rama/commit.
Nombre sugerido: `reports/figures/entrega3/pytest_suite.png`.

## 3. Calidad de datos — Deepchecks

Se requieren los datos procesados del proyecto; las pruebas de PyTest no los
sustituyen. Con los CSV y Parquet preparados, ejecutar por pregunta:

```bash
uv run --no-project --python 3.12 \
  --with-requirements requirements-quality.txt \
  python -m proyecto_mibici.data_quality --question p1
uv run --no-project --python 3.12 \
  --with-requirements requirements-quality.txt \
  python -m proyecto_mibici.data_quality --question p2
uv run --no-project --python 3.12 \
  --with-requirements requirements-quality.txt \
  python -m proyecto_mibici.data_quality --question p3
```

Cada ejecución genera `reports/data_quality/deepchecks_<pregunta>.json` y
`reports/data_quality/deepchecks_<pregunta>.html`. El reporte incluye tipos,
faltantes y duplicados (preflight), además de la suite tabular y drift.

La **guía del equipo (30-oct-2026)** documenta estos resultados preliminares:

| Pregunta | Hallazgos documentados por el equipo | Comprobación pendiente |
|---|---|---|
| P1 | Sin faltantes ni duplicados; drift multivariado bajo (AUC 0.60–0.62) | Abrir JSON/HTML final y confirmar |
| P2 | Sin faltantes ni duplicados; drift multivariado bajo (AUC 0.60–0.62) | Abrir JSON/HTML final y confirmar |
| P3 | Drift en 5 de 336 columnas (AUC 0.74); 22 columnas de madrugada sin variación | Revisar columnas y condición en HTML final |

**Evidencias disponibles:** capturas reales de los tres HTML, mostrando el
resultado de los chequeos y los hallazgos. Capturas generadas:
`deepchecks_p1.png`, `deepchecks_p2.png`, `deepchecks_p3.png`.
Los HTML/JSON generados están excluidos de Git: compartir capturas agregadas,
no archivos con datos sensibles.

## 4. Seguridad y dependencias

```bash
# Análisis estático de código Python.
uv run --frozen bandit -r proyecto_mibici -q
echo $?

# Auditoría de dependencias del entorno principal.
uv run --frozen pip-audit -r requirements.txt --no-deps --disable-pip

# Auditoría del entorno aislado de Deepchecks (riesgos aceptados documentados).
uv run --no-project --python 3.12 --with pip-audit==2.10.1 \
  pip-audit -r requirements-quality.txt \
  --ignore-vuln PYSEC-2026-3447 --ignore-vuln GHSA-xmj3-9gf7-h52w
```

En macOS/Linux los tres controles se pueden ejecutar en conjunto mediante
`make security`. Revisa los fundamentos de los hallazgos y excepciones en
[`seguridad_entrega3.md`](seguridad_entrega3.md).

**Estado verificado en las ejecuciones locales de la Entrega 3:**

- Bandit terminó sin hallazgos (`exit code 0`).
- pip-audit del proyecto **no pasó**: detectó 4 vulnerabilidades conocidas
  en `pyasn1==0.6.4` (CVE-2026-119194, CVE-2026-119195, CVE-2026-119196 y
  CVE-2026-119197); la corrección indicada por la auditoría es `0.6.5`.
- La auditoría aislada de Deepchecks **se ejecutó** y reportó `No known vulnerabilities found, 3 ignored`. Las tres vulnerabilidades ignoradas no están corregidas y sus excepciones deben mantenerse explícitas y justificadas.

**No describir el control como exitoso** mientras permanezcan vulnerabilidades
sin resolver. Después de actualizar `uv.lock` y regenerar `requirements.txt`,
repetir la auditoría y registrar el resultado nuevo, sin borrar la evidencia de
la detección original. Nunca ignorar las CVE de `pyasn1` solo para obtener un
resultado verde.

**Evidencias disponibles:** `bandit_sin_hallazgos.png`, `pip_audit_vulnerabilidades_detectadas.png` y
`pip_audit_deepchecks.png`, cada una tomada en la versión final del código.
Si hay un fallo, capturar el hallazgo y señalar el seguimiento.

## 5. Capturas y publicación

En macOS, presionar **Command + Shift + 4** y seleccionar el área de la
terminal o el navegador con el resultado visible. Guardar los PNG en
`reports/figures/entrega3/` y enlazarlos desde el README **solo después de
crearlos y revisar que no contengan datos personales, tokens o rutas sensibles**.

Antes del commit:

```bash
git status --short
git diff --check
git add README.md docs/docs/evidencias_entrega3.md reports/figures/entrega3/
git diff --cached --stat
git diff --cached --check
```

No incluir `data/data_manifest.json`, `.env`, bases ni artefactos locales. Abrir
un Pull Request hacia `main`, solicitar revisión de un compañero y usar merge
commit según la guía colaborativa.

## 6. Checklist de verificación final

- [ ] PyTest ejecutado en el commit final y captura real guardada.
- [ ] Reportes P1, P2 y P3 contrastados con las cifras del README y capturados.
- [ ] Bandit y ambos pip-audit ejecutados; vulnerabilidades corregidas o documentadas.
- [ ] Evidencias PNG revisadas para evitar información sensible.
- [ ] README actualizado con enlaces a las capturas que sí existen.
- [ ] PR individual revisado por otro integrante y fusionado a través de GitHub.
