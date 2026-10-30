# Entrega 3 — Chequeos de seguridad básicos (punto 3)

Revisión del repositorio para confirmar que no expone credenciales, contraseñas ni datos sensibles, y
análisis automatizado del código (bandit) y de las dependencias (pip-audit). Fecha de la revisión:
2026-10-30, sobre `main` en `4b70726` (después del PR #17).

## 1. Cómo ejecutar los chequeos

Con make (Linux/macOS o Windows con make instalado):

```bash
make security
```

Sin make, los mismos tres comandos:

```bash
uv sync --frozen
uv run --frozen bandit -r proyecto_mibici -q
uv run --frozen pip-audit -r requirements.txt --no-deps --disable-pip
uv run --no-project --python 3.12 --with pip-audit==2.10.1 \
  pip-audit -r requirements-quality.txt \
  --ignore-vuln PYSEC-2026-3447 --ignore-vuln GHSA-xmj3-9gf7-h52w
```

- `bandit` analiza el código del paquete `proyecto_mibici`.
- El primer `pip-audit` revisa las dependencias exactas del proyecto (`requirements.txt`, exportado de
  `uv.lock`, que es también el entorno de `mlflow run`).
- El segundo revisa el entorno aislado de Deepchecks (`requirements-quality.txt`, Python 3.12). Las dos
  vulnerabilidades ignoradas se explican en la sección 4.

Las versiones de bandit (1.9.4) y pip-audit (2.10.1) están fijas en el grupo `dev` de `pyproject.toml`.

## 2. Resultados

| Chequeo | Antes | Después |
|---|---|---|
| bandit | 7 hallazgos (2 medios B310, 5 bajos B404/B603/B607) | 0 hallazgos (3 supresiones justificadas: B310, B404, B603) |
| pip-audit, proyecto | 2 vulnerabilidades (PYSEC-2026-113 en pyarrow 19.0.1, PYSEC-2026-2270 en python-dotenv 1.0.1), más 3 CVE en tornado 6.5.10 que no se veían por `requirements.txt` desfasado | 0 vulnerabilidades |
| pip-audit, entorno de Deepchecks | 2 vulnerabilidades en setuptools 80.10.2 | 0 vulnerabilidades; 2 aceptadas y documentadas (pip-audit las reporta como "3 ignored" porque lista una dos veces) |
| Secretos en archivos versionados (detect-secrets 1.5.0) | 4 avisos, todos falsos positivos | — |
| Secretos en el historial de git (todas las ramas) | 0 coincidencias | — |

## 3. Correcciones al código y a las dependencias

**bandit**

- B310 (`urlopen` con esquemas arbitrarios, severidad media): toda petición de red pasa por
  `dataset._solicitud`, que solo acepta HTTPS de `mibici.net`/`www.mibici.net`; `dataset._abrir` es el único
  lugar con `urlopen`. Antes, `descubrir_fuentes` abría la página de datos sin esa validación. La prueba
  `test_requests_outside_official_https_domain_are_rejected` cubre `http://`, `file://`, un dominio parecido y
  uno ajeno, y confirma que no se abre conexión ni se crea archivo.
- B607 (ejecutable con ruta parcial): `tracking._git` resuelve git con `shutil.which` y falla con un mensaje
  claro si no está instalado.
- B404/B603 (uso de `subprocess`): se suprimen con `# nosec` porque solo se ejecuta git, sin shell y con
  argumentos fijos del propio código; no entra ningún dato del usuario.

**pip-audit**

| Paquete | Versión anterior | Versión nueva | Observación |
|---|---|---|---|
| pyarrow | 19.0.1 | 23.0.1 | Misma `data_version` (`09a6a584dd5d`) y mismos conteos en la ingesta real; Deepchecks sin cambios |
| python-dotenv | 1.0.1 | 1.2.2 | — |
| tornado | 6.5.10 | 6.6 | Dependencia transitiva de `ipykernel` |

`requirements.txt` estaba desfasado de `uv.lock` (faltaban 24 paquetes que llegaron con `ipykernel`), por eso
la primera auditoría no detectaba tornado. Se volvió a exportar con:

```bash
uv export --frozen --no-dev --no-emit-project --format requirements-txt --output-file requirements.txt
```

## 4. Riesgo aceptado

`setuptools` 80.10.2 en el entorno de Deepchecks (PYSEC-2026-3447, corregido en 83.0.0; GHSA-xmj3-9gf7-h52w,
corregido en 84.1.0). Deepchecks 0.19.1 importa `pkg_resources`, que las versiones corregidas eliminan: con
setuptools 84.1 la validación falla con `No module named 'pkg_resources'`. Se acepta porque ese entorno es
aislado (`uv run --no-project`), solo valida datos agregados y no se usa en `mlflow run`, en el Model Registry
ni en la inferencia. Se revisará cuando Deepchecks publique una versión sin `pkg_resources`.

## 5. Credenciales y datos sensibles

- **Variables de entorno:** el proyecto no necesita credenciales. La única variable es `MLFLOW_TRACKING_URI`;
  `config.py` la lee con `os.getenv` y carga un `.env` local con python-dotenv. `.env.example` documenta las
  variables (incluidas las de autenticación de MLflow, comentadas y vacías) y es el único archivo `.env*` que
  se versiona.
- **`.gitignore`** excluye: `.env` y `.env.*`; los datos de MiBici en `data/` (solo se versiona
  `data/data_manifest.json`, sin filas de viajes); `mlflow.db`, `mlruns/` y `mlartifacts/`; modelos serializados
  (`*.pkl`, `*.joblib`, `*.skops`, `*.onnx`, entre otros); llaves y credenciales (`*.pem`, `*.key`, `*.p12`,
  `*.pfx`, `credentials*.json`, `service-account*.json`, `.netrc`); y los reportes generados de Deepchecks.
  Ningún archivo versionado queda dentro de esas reglas.
- **Historial de git:** no contiene llaves privadas, tokens de GitHub, AWS o Slack, ni asignaciones de
  contraseñas, y nunca se versionó un `.env`, `.pem`, `.key` ni archivo de credenciales.
- **detect-secrets** sobre los archivos versionados marca 4 archivos, todos falsos positivos: hashes sha256 del
  manifiesto de datos, un hash de commit ficticio en `tests/test_model_registry.py` e imágenes PNG incrustadas
  en dos notebooks.
- **Privacidad en MLflow:** la base común excluye `usuario_id`, `genero` y `anio_nacimiento`, y los modelos se
  registran con `input_example` agregado por estación (ver `registro_modelos_entrega2.md`).

## 6. Hallazgo pendiente

`notebooks/1.1-AAAA-EDA_P2.ipynb` conserva en las salidas de una celda filas de viajes con `usuario_id`,
`genero` y `anio_nacimiento`. La limpieza (borrar las salidas del notebook) queda asignada al autor del
notebook en el siguiente PR. Aunque se limpie, la versión anterior permanece en el historial de `main`;
eliminarla de verdad requiere reescribir el historial y es una decisión del equipo.
