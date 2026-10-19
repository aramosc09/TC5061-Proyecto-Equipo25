from pathlib import Path
import json
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from loguru import logger
import typer

from proyecto_mibici.features_p2 import FeatureEngineeringP2
from proyecto_mibici.config import (INTERIM_DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, 
                                    DEFAULT_RANDOM_STATE, DEFAULT_DESFASE_HORAS, DEFAULT_PERCENTIL_UMBRAL,
                                    DEFAULT_FECHA_CORTE_SPLIT)

app = typer.Typer()

# ==============================================================================
# PIPELINE ENVOLVENTE (WRAPPER PARA PRODUCCIÓN)
# ==============================================================================
class PipelineRiesgoP2:
    """
    Empaqueta el FeatureEngineeringP2 y el Modelo de Clasificación 
    en una sola entidad para garantizar la portabilidad en MLOps.
    """
    def __init__(self, generador_features, modelo, tipo_modelo: str):
        self.generador_features = generador_features
        self.modelo = modelo
        self.tipo_modelo = tipo_modelo
        self.columnas_drop = ['tiempo', 'estacion_id', 'target_riesgo', 'B', 'b']
        self.label_encoder = LabelEncoder() if tipo_modelo == 'xgboost' else None

    def fit(self, df_viajes, df_nom):
        """Ajusta tanto el generador de características como el modelo de clasificación."""
        train_features = self.generador_features.fit(df_viajes, df_nom).transform(df_viajes, df_nom)
        
        X_train = train_features.drop(columns=self.columnas_drop)
        y_train = train_features['target_riesgo']
        
        if self.tipo_modelo == 'xgboost':
            y_train = self.label_encoder.fit_transform(y_train)
            
        self.modelo.fit(X_train, y_train)
        return self

    def predict(self, df_viajes, df_nom):
        """Transforma datos nuevos e infiere las clases de riesgo de negocio (-1, 0, 1)."""
        test_features = self.generador_features.transform(df_viajes, df_nom)
        X_test = test_features.drop(columns=self.columnas_drop)
        
        predicciones_raw = self.modelo.predict(X_test)
        
        if self.tipo_modelo == 'xgboost':
            return self.label_encoder.inverse_transform(predicciones_raw)
        return predicciones_raw

    def transform_features(self, df_viajes, df_nom):
        """Método auxiliar para obtener las matrices procesadas del generador."""
        features = self.generador_features.transform(df_viajes, df_nom)
        X = features.drop(columns=self.columnas_drop)
        y = features['target_riesgo']
        return X, y, features


# ==============================================================================
# CLASE ORQUESTADORA DE ENTRENAMIENTO
# ==============================================================================
class EntrenadorModeloP2:
    """
    Orquestador que entrena el pipeline unificado, exporta parquets procesados y registra en MLFlow.
    """
    
    def __init__(self, tipo_modelo: str, fecha_corte: str, ventana_prediccion: int = 2, percentil_umbral: float = 0.90, random_state: int = 42, **kwargs):
        self.tipo_modelo = tipo_modelo.lower()
        self.fecha_corte = pd.to_datetime(fecha_corte)
        self.ventana_prediccion = ventana_prediccion
        self.percentil_umbral = percentil_umbral
        self.random_state = random_state
        self.kwargs = kwargs
        
        self.metrics = {}
        self.matriz_path = f"P2_confusion_matrix_{self.tipo_modelo}.png"
        
        # Instanciar componentes internos
        generador = FeatureEngineeringP2(
            ventana_prediccion=self.ventana_prediccion, 
            percentil_umbral=self.percentil_umbral
        )
        modelo_base = self._construir_modelo()
        
        # Empaquetar componentes en el Pipeline Unificado
        self.pipeline = PipelineRiesgoP2(generador, modelo_base, self.tipo_modelo)

    def _construir_modelo(self):
        """Instancia el modelo seleccionado asegurando la semilla random_state."""
        if self.tipo_modelo == 'random_forest':
            return RandomForestClassifier(
                n_estimators=self.kwargs.get('n_estimators', 100),
                max_depth=self.kwargs.get('max_depth', 10),
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.tipo_modelo == 'logistic_regression':
            return LogisticRegression(
                C=self.kwargs.get('c_reg', 1.0),
                class_weight='balanced',
                max_iter=1000,
                random_state=self.random_state
            )
        elif self.tipo_modelo == 'xgboost':
            return XGBClassifier(
                n_estimators=self.kwargs.get('n_estimators', 100),
                max_depth=self.kwargs.get('max_depth', 6),
                learning_rate=self.kwargs.get('learning_rate', 0.1),
                random_state=self.random_state,
                eval_metric='mlogloss',
                n_jobs=-1
            )
        else:
            raise ValueError(f"Modelo '{self.tipo_modelo}' no soportado.")

    def cargar_y_particionar(self, ruta_viajes: Path, ruta_nom: Path):
        """Carga los datos y realiza el split Out-of-Time."""
        df_viajes = pd.read_parquet(ruta_viajes)
        df_nom = pd.read_csv(ruta_nom)
        
        if 'id' in df_nom.columns:
            df_nom.rename(columns={'id': 'estacion_id', 'dpcapacity': 'capacidad'}, inplace=True)
            
        df_viajes['inicio_viaje'] = pd.to_datetime(df_viajes['inicio_viaje'])
        df_viajes = df_viajes.sort_values('inicio_viaje')
        
        viajes_train = df_viajes[df_viajes['inicio_viaje'] < self.fecha_corte].copy()
        viajes_test = df_viajes[df_viajes['inicio_viaje'] >= self.fecha_corte].copy()
        
        return viajes_train, viajes_test, df_nom

    def procesar_y_guardar_datasets(self, viajes_train: pd.DataFrame, viajes_test: pd.DataFrame, df_nom: pd.DataFrame):
        """Ajusta el generador con train, transforma ambos splits y guarda los .parquet procesados."""
        logger.info("   * Generando características respetando el split Out-of-Time...")
        
        # Fit exclusivo en Train para evitar data leakage
        self.pipeline.generador_features.fit(viajes_train, df_nom)
        
        train_features = self.pipeline.generador_features.transform(viajes_train, df_nom)
        test_features = self.pipeline.generador_features.transform(viajes_test, df_nom)
        
        # Guardar en data/processed/
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        ruta_train_parquet = PROCESSED_DATA_DIR / "p2_train_features.parquet"
        ruta_test_parquet = PROCESSED_DATA_DIR / "p2_test_features.parquet"
        
        train_features.to_parquet(ruta_train_parquet, index=False)
        test_features.to_parquet(ruta_test_parquet, index=False)
        
        # Conteos y Métricas de Muestras
        total_crudos = len(viajes_train) + len(viajes_test)
        total_procesados = len(train_features) + len(test_features)
        
        logger.info("\n" + "=" * 65)
        logger.info("📊 RESUMEN Y CONTEO DE REGISTROS:")
        logger.info(f"   • Viajes Crudos Totales : {total_crudos:,}")
        logger.info(f"     - Train Crudo         : {len(viajes_train):,} ({len(viajes_train)/total_crudos:.1%})")
        logger.info(f"     - Test Crudo          : {len(viajes_test):,} ({len(viajes_test)/total_crudos:.1%})")
        logger.info(f"   • Malla Temporal Procesada Totales : {total_procesados:,}")
        logger.info(f"     - Train Procesado (Filas)       : {len(train_features):,} ({len(train_features)/total_procesados:.1%})")
        logger.info(f"     - Test Procesado  (Filas)       : {len(test_features):,} ({len(test_features)/total_procesados:.1%})")
        logger.info("=" * 65)
        logger.success(f"💾 Archivos parquet exportados exitosamente a:\n   • {ruta_train_parquet}\n   • {ruta_test_parquet}")
        
        return train_features, test_features

    def ejecutar_pipeline(self, ruta_viajes: Path, ruta_nom: Path, ruta_manifest: Path, ruta_salida_modelo: Path, run_name: str, solo_procesar: bool = False):
        logger.info(f"   * Cargando datos de entrada...")
        viajes_train, viajes_test, df_nom = self.cargar_y_particionar(ruta_viajes, ruta_nom)
        
        # Procesar características y exportar DataFrames a data/processed/
        train_features, test_features = self.procesar_y_guardar_datasets(viajes_train, viajes_test, df_nom)
        
        if solo_procesar:
            logger.info("🛑 Deteniendo ejecución antes del entrenamiento (--solo-procesar activado).")
            return
        
        X_train = train_features.drop(columns=self.pipeline.columnas_drop)
        y_train = train_features['target_riesgo']
        X_test = test_features.drop(columns=self.pipeline.columnas_drop)
        y_test = test_features['target_riesgo']
        
        mlflow.set_experiment("MiBici_Riesgo_Inventario_P2")
        
        with mlflow.start_run(run_name=run_name):
            logger.info(f"   * Entrenando modelo: {self.tipo_modelo}...")
            
            if self.tipo_modelo == 'xgboost':
                y_train_fit = self.pipeline.label_encoder.fit_transform(y_train)
            else:
                y_train_fit = y_train
                
            self.pipeline.modelo.fit(X_train, y_train_fit)
            
            # Registrar hiperparámetros globales, del modelo y métricas de tamaño de datos
            mlflow.log_params({
                "tipo_modelo": self.tipo_modelo,
                "fecha_corte": self.fecha_corte.strftime("%Y-%m-%d"),
                "ventana_prediccion": self.ventana_prediccion,
                "percentil_umbral": self.percentil_umbral,
                "random_state": self.random_state,
                "umbral_calculado_b": self.pipeline.generador_features.u_val_,
                "n_viajes_train": len(viajes_train),
                "n_viajes_test": len(viajes_test),
                "n_samples_train_procesados": len(X_train),
                "n_samples_test_procesados": len(X_test)
            })
            mlflow.log_params(self.kwargs)
            
            # Registrar el Manifiesto de Datos en MLFlow
            if ruta_manifest.exists():
                logger.info(f"   * Registrando manifiesto de datos ({ruta_manifest}) en MLFlow...")
                mlflow.log_artifact(str(ruta_manifest), artifact_path="metadata")
            else:
                logger.warning(f"   ! No se encontró el manifiesto de datos en: {ruta_manifest}")

            logger.info("   * Generando inferencias en el conjunto de prueba...")
            predicciones_raw = self.pipeline.modelo.predict(X_test)
            
            if self.tipo_modelo == 'xgboost':
                predicciones = self.pipeline.label_encoder.inverse_transform(predicciones_raw)
            else:
                predicciones = predicciones_raw
            
            self.metrics['accuracy'] = accuracy_score(y_test, predicciones)
            self.metrics['macro_precision'] = precision_score(y_test, predicciones, average='macro', zero_division=0)
            self.metrics['macro_recall'] = recall_score(y_test, predicciones, average='macro', zero_division=0)
            self.metrics['macro_f1'] = f1_score(y_test, predicciones, average='macro', zero_division=0)
            
            mlflow.log_metrics(self.metrics)
            
            print(f"\n--- REPORTE DE CLASIFICACIÓN ({run_name}) ---")
            print(classification_report(y_test, predicciones, labels=[-1, 0, 1], 
                                        target_names=['Sin Bicis (-1)', 'Normal (0)', 'Sin Anclajes (1)']))
            
            # Generación de Matriz de Confusión
            plt.figure(figsize=(6, 5))
            sns.heatmap(confusion_matrix(y_test, predicciones, labels=[-1, 0, 1]), annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Sin Bicis (-1)', 'Normal (0)', 'Sin Anclajes (1)'], 
                        yticklabels=['Sin Bicis (-1)', 'Normal (0)', 'Sin Anclajes (1)'])
            plt.title(f'Matriz de Confusión - {self.tipo_modelo.upper()}')
            plt.ylabel('Valor Real')
            plt.xlabel('Predicción')
            plt.tight_layout()
            plt.savefig(self.matriz_path)
            plt.close()
            
            mlflow.log_artifact(self.matriz_path)
            
            # Guardado Unificado
            self.guardar_pipeline(ruta_salida_modelo)
            mlflow.log_artifact(str(ruta_salida_modelo / f"pipeline_{self.tipo_modelo}_p2.joblib"))
            
        logger.success(f"   ✓ Experimento '{run_name}', Manifiesto, Parquets y Pipeline unificado registrados correctamente.")

    def guardar_pipeline(self, ruta_directorio: Path):
        """Guarda todo el objeto pipeline en un solo archivo serializado."""
        ruta_directorio.mkdir(parents=True, exist_ok=True)
        ruta_archivo = ruta_directorio / f"pipeline_{self.tipo_modelo}_p2.joblib"
        joblib.dump(self.pipeline, ruta_archivo)


# ==============================================================================
# EJECUCIÓN CLI
# ==============================================================================
@app.command()
def main(
    # --- Parámetros de Rutas y Datos ---
    viajes_path: Path = INTERIM_DATA_DIR / "viajes_unificados.parquet",
    nomenclatura_path: Path = RAW_DATA_DIR / "nomenclatura_2026_09.csv",
    manifest_path: Path = Path("data/data_manifest.json"),
    models_path: Path = MODELS_DIR,
    
    # --- Selección del Modelo y Banderas ---
    modelo: str = typer.Option("random_forest", help="Algoritmo: 'random_forest', 'logistic_regression' o 'xgboost'"),
    random_state: int = typer.Option(DEFAULT_RANDOM_STATE, help="Semilla aleatoria global para reproducibilidad"),
    fecha_corte: str = typer.Option(DEFAULT_FECHA_CORTE_SPLIT, help="Fecha para el split temporal YYYY-MM-DD"),
    solo_procesar: bool = typer.Option(False, "--solo-procesar", help="Solo genera y guarda parquets de train/test sin entrenar modelos"),
    
    # --- Hiperparámetros de Negocio / Preprocesamiento ---
    percentil_umbral: float = typer.Option(DEFAULT_PERCENTIL_UMBRAL, help="Percentil para calcular el umbral u de riesgo"),
    ventana_prediccion: int = typer.Option(DEFAULT_DESFASE_HORAS, help="Ventana de tiempo en horas a predecir"),
    
    # --- Hiperparámetros del Algoritmo ---
    n_estimators: int = typer.Option(100, help="[RF / XGBoost] Número de árboles"),
    max_depth: int = typer.Option(10, help="[RF / XGBoost] Profundidad máxima"),
    learning_rate: float = typer.Option(0.1, help="[XGBoost] Tasa de aprendizaje"),
    c_reg: float = typer.Option(1.0, help="[Logistic Regression] Inverso de regularización C")
):
    """
    Entrena un Pipeline de Machine Learning registrando los hiperparámetros, el manifiesto de datos,
    los parquets procesados y la semilla aleatoria en MLFlow.
    """
    logger.info(f"Iniciando ejecucion para modelo: {modelo.upper()} (random_state={random_state})...")
    
    kwargs = {}
    if modelo == "random_forest":
        kwargs = {"n_estimators": n_estimators, "max_depth": max_depth}
    elif modelo == "xgboost":
        kwargs = {"n_estimators": n_estimators, "max_depth": max_depth, "learning_rate": learning_rate}
    elif modelo == "logistic_regression":
        kwargs = {"c_reg": c_reg}
    else:
        raise ValueError(f"Modelo '{modelo}' no válido.")

    timestamp = datetime.now().strftime("%H%M%S")
    run_name = f"{modelo}_rs{random_state}_{timestamp}"

    entrenador = EntrenadorModeloP2(
        tipo_modelo=modelo,
        fecha_corte=fecha_corte,
        ventana_prediccion=ventana_prediccion,
        percentil_umbral=percentil_umbral,
        random_state=random_state,
        **kwargs
    )
    
    entrenador.ejecutar_pipeline(
        ruta_viajes=viajes_path,
        ruta_nom=nomenclatura_path,
        ruta_manifest=manifest_path,
        ruta_salida_modelo=models_path,
        run_name=run_name,
        solo_procesar=solo_procesar
    )

if __name__ == "__main__":
    app()