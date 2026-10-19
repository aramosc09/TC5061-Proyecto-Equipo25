.PHONY: requirements data features train ui test lint

requirements:
	uv sync --frozen

## Ingesta: CSV crudos -> Parquet unificado + manifiesto de datos
data:
	uv run --frozen python -m proyecto_mibici.dataset procesar-datos $(ARGS)

## Features estacion-hora de P1/P2 y perfiles de P3
features:
	uv run --frozen python -m proyecto_mibici.features $(ARGS)

## Baselines (y candidatos con --candidates) registrados en MLflow
train:
	uv run --frozen python -m proyecto_mibici.modeling.train $(ARGS)

ui:
	uv run --frozen mlflow ui --backend-store-uri sqlite:///mlruns/tracking.db

test:
	uv run --frozen pytest -q

lint:
	uv run --frozen ruff check proyecto_mibici
