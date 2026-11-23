#################################################################################
# GLOBALS                                                                       #
#################################################################################

PYTHON_INTERPRETER = uv run --frozen python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync --frozen
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	uv run --frozen ruff check proyecto_mibici tests

## Format source code with ruff
.PHONY: format
format:
	uv run --frozen ruff check proyecto_mibici tests --fix
	uv run --frozen ruff format proyecto_mibici tests



## Run tests
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests


# setuptools<81 en el entorno de calidad: Deepchecks 0.19.1 importa pkg_resources, que las versiones
# corregidas eliminan. Riesgo aceptado: ese entorno solo valida datos y no entra a mlflow run.
QUALITY_ACCEPTED_VULNS = --ignore-vuln PYSEC-2026-3447 --ignore-vuln GHSA-xmj3-9gf7-h52w

## Scan code with bandit and dependencies with pip-audit
.PHONY: security
security:
	uv run --frozen bandit -r proyecto_mibici -q
	uv run --frozen pip-audit -r requirements.txt --no-deps --disable-pip
	uv run --no-project --python 3.12 --with pip-audit==2.10.1 \
		pip-audit -r requirements-quality.txt $(QUALITY_ACCEPTED_VULNS)


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv sync --frozen
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) -m proyecto_mibici.dataset procesar-datos

## Build shared station-hour features
.PHONY: features
features:
	$(PYTHON_INTERPRETER) -m proyecto_mibici.experiments features

## Run all three baselines through MLflow Projects
.PHONY: experiment
experiment:
	$(PYTHON_INTERPRETER) -m proyecto_mibici.experiments ejecutar-proyecto

## Run a reproducible Locust load test against the API
.PHONY: load-test
load-test:
	mkdir -p reports/figures/entrega6 reports/monitoreo/carga
	LOCUST_BATCH_SIZE=$${LOCUST_BATCH_SIZE:-10} \
	LOCUST_SEED=$${LOCUST_SEED:-42} \
	uv run --frozen --extra carga locust -f locustfile.py \
	--host $${LOCUST_HOST:-http://127.0.0.1:8000} \
	--users $${LOCUST_USERS:-20} \
	--spawn-rate $${LOCUST_SPAWN_RATE:-5} \
	--run-time $${LOCUST_RUN_TIME:-60s} \
	--headless \
	--html reports/figures/entrega6/locust_report.html \
	--csv reports/monitoreo/carga/locust

## Monitor CPU and memory while Compose is under load
.PHONY: monitor-resources
monitor-resources:
	uv run --frozen python -m proyecto_mibici.monitoreo_recursos \
		--duracion $${MONITOR_DURATION:-60} \
		--intervalo $${MONITOR_INTERVAL:-1} \
		--runtime $${MONITOR_RUNTIME:-auto}

## Generar los tres JSON reales para Locust
.PHONY: load-peticiones
load-peticiones:
	uv run --frozen python -m pruebas_carga.generar_peticiones

## Ejecutar cuatro escalones con uso de recursos simultáneo
.PHONY: load-escalones
load-escalones:
	uv run --frozen python -m pruebas_carga.ejecutar --runtime $${MONITOR_RUNTIME:-podman} \
		--prefijo $${CARGA_PREFIJO:-base_1cpu} --tipo $${LOCUST_TIPO:-mixto} \
		--escalones $${LOCUST_ESCALONES:-10:120,25:120,50:120,100:120}

## Estimar costo mensual por inferencia según las mediciones de Locust
.PHONY: finops
finops:
	uv run --frozen python -m proyecto_mibici.finops

## Verify recorded runs and artifacts
.PHONY: verify
verify:
	$(PYTHON_INTERPRETER) -m proyecto_mibici.experiments verificar-experimento


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
