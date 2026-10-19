from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin

from loguru import logger
from tqdm import tqdm
import typer

# Ajusta las importaciones de tu config según la estructura de Cookiecutter
from proyecto_mibici.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR, RAW_DATA_DIR, DEFAULT_DESFASE_HORAS, DEFAULT_PERCENTIL_UMBRAL

app = typer.Typer()

# ==============================================================================
# CLASE TRANSFORMADORA DE CARACTERÍSTICAS
# ==============================================================================
class FeatureEngineeringP2(BaseEstimator, TransformerMixin):
    """
    Transformador para generar las características temporales y la variable objetivo (clase de riesgo)
    del Problema 2, asegurando una malla temporal continua.
    Incluye utilidades de visualización para ejecutar desde el pipeline de entrenamiento.
    """
    
    def __init__(self, ventana_prediccion = DEFAULT_DESFASE_HORAS, percentil_umbral = DEFAULT_PERCENTIL_UMBRAL):
        self.ventana_prediccion = ventana_prediccion
        self.percentil_umbral = percentil_umbral
        self.u_val_ = None
        
    def _agrupar_viajes(self, df_viajes, df_nomenclatura):
        """Genera una serie de tiempo continua hora a hora para cada estación."""
        df = df_viajes.copy()
        df['inicio_viaje'] = pd.to_datetime(df['inicio_viaje'])
        df['fin_viaje'] = pd.to_datetime(df['fin_viaje'])

        tiempo_minimo = df['inicio_viaje'].dt.floor('h').min()
        tiempo_maximo = df['inicio_viaje'].dt.floor('h').max()
        rango_horas = pd.date_range(start=tiempo_minimo, end=tiempo_maximo, freq='1h')
        
        estaciones_totales = df_nomenclatura['estacion_id'].unique()
        
        grid_completo = pd.MultiIndex.from_product(
            [rango_horas, estaciones_totales], 
            names=['tiempo', 'estacion_id']
        ).to_frame(index=False)

        salidas = df.groupby([df['inicio_viaje'].dt.floor('h'), 'origen_id']).size().reset_index(name='salidas')
        salidas.rename(columns={'inicio_viaje': 'tiempo', 'origen_id': 'estacion_id'}, inplace=True)

        llegadas = df.groupby([df['fin_viaje'].dt.floor('h'), 'destino_id']).size().reset_index(name='llegadas')
        llegadas.rename(columns={'fin_viaje': 'tiempo', 'destino_id': 'estacion_id'}, inplace=True)

        df_agrupado = pd.merge(grid_completo, salidas, on=['tiempo', 'estacion_id'], how='left')
        df_agrupado = pd.merge(df_agrupado, llegadas, on=['tiempo', 'estacion_id'], how='left')
        
        df_agrupado['salidas'] = df_agrupado['salidas'].fillna(0)
        df_agrupado['llegadas'] = df_agrupado['llegadas'].fillna(0)

        return df_agrupado.sort_values(['estacion_id', 'tiempo'])

    def fit(self, df_viajes, df_nomenclatura):
        """Calcula y congela el umbral 'u_val_'."""
        df_eda = self._agrupar_viajes(df_viajes, df_nomenclatura)
        
        df_eda['llegadas_ventana'] = df_eda.groupby('estacion_id')['llegadas'].transform(lambda x: x.rolling(self.ventana_prediccion).sum().shift(-1))
        df_eda['salidas_ventana'] = df_eda.groupby('estacion_id')['salidas'].transform(lambda x: x.rolling(self.ventana_prediccion).sum().shift(-1))
        df_eda['B'] = df_eda['llegadas_ventana'] - df_eda['salidas_ventana']
        
        df_eda = df_eda.merge(df_nomenclatura[['estacion_id', 'capacidad']], on='estacion_id', how='left')
        df_eda['b'] = df_eda['B'] / df_eda['capacidad']
        
        self.u_val_ = df_eda['b'].abs().quantile(self.percentil_umbral)
        
        return self

    def transform(self, df_viajes, df_nomenclatura):
        """Aplica el umbral calculado y extrae las características temporales."""
        if self.u_val_ is None:
            raise ValueError("El transformador debe ser ajustado con fit() antes de llamar a transform().")
            
        df_features = self._agrupar_viajes(df_viajes, df_nomenclatura)
        
        df_features['llegadas_ventana'] = df_features.groupby('estacion_id')['llegadas'].transform(lambda x: x.rolling(self.ventana_prediccion).sum().shift(-1))
        df_features['salidas_ventana'] = df_features.groupby('estacion_id')['salidas'].transform(lambda x: x.rolling(self.ventana_prediccion).sum().shift(-1))
        df_features['B'] = df_features['llegadas_ventana'] - df_features['salidas_ventana']
        
        df_features = df_features.merge(df_nomenclatura[['estacion_id', 'capacidad']], on='estacion_id', how='left')
        df_features['b'] = df_features['B'] / df_features['capacidad']
        
        df_features['hora'] = df_features['tiempo'].dt.hour
        df_features['dia_semana'] = df_features['tiempo'].dt.dayofweek
        df_features['es_fin_semana'] = (df_features['tiempo'].dt.dayofweek >= 5).astype(int)
        
        condiciones = [
            (df_features['b'] <= -self.u_val_),
            (df_features['b'] >= self.u_val_)
        ]
        df_features['target_riesgo'] = np.select(condiciones, [-1, 1], default=0)
        
        df_features.dropna(subset=['b', 'target_riesgo'], inplace=True)
        
        columnas_finales = [
            'tiempo', 'estacion_id', 'hora', 'dia_semana', 'es_fin_semana', 
            'salidas', 'llegadas', 'capacidad', 'B', 'b', 'target_riesgo'
        ]
        
        return df_features[columnas_finales].reset_index(drop=True)

    def graficar_eda(self, df_features, guardar_directorio=None):
        """
        Genera gráficas de Análisis Exploratorio sobre el DataFrame transformado.
        
        :param df_features: DataFrame resultante del método transform()
        :param guardar_directorio: Objeto Path o str indicando la carpeta donde guardar los .png (opcional)
        """
        if self.u_val_ is None:
            raise ValueError("Debes hacer fit() al transformador antes de graficar.")
            
        if guardar_directorio:
            ruta_dir = Path(guardar_directorio)
            ruta_dir.mkdir(parents=True, exist_ok=True) # Asegura que la carpeta exista
            
        sns.set_theme(style="whitegrid")
        df_plot = df_features.copy()
        
        # 1. Histograma del Balance Normalizado
        plt.figure(figsize=(10, 6))
        sns.histplot(df_plot['b'], bins=100, kde=True, color='steelblue')
        plt.axvline(x=-self.u_val_, color='red', linestyle='--', label=f'-u (Riesgo sin bicis: {-self.u_val_:.2f})')
        plt.axvline(x=self.u_val_, color='orange', linestyle='--', label=f'+u (Riesgo sin anclajes: {self.u_val_:.2f})')
        plt.title('Distribución del Balance Normalizado (b)', fontsize=14)
        plt.xlabel('Balance Normalizado (b = B / capacidad)')
        plt.ylabel('Frecuencia')
        plt.legend()
        if guardar_directorio:
            plt.savefig(ruta_dir / "P2_distribucion_balance.png", bbox_inches='tight')
        plt.show()

        # 2. Mapa de Calor (Desbalance Promedio)
        dias_map = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}
        df_plot['dia_nombre'] = df_plot['dia_semana'].map(dias_map)
        heatmap_data = df_plot.groupby(['dia_nombre', 'hora'])['b'].apply(lambda x: x.abs().mean()).unstack()
        heatmap_data = heatmap_data.reindex(list(dias_map.values()))
        
        plt.figure(figsize=(14, 6))
        sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False)
        plt.title('Intensidad Promedio del Desbalance (|b|) por Día y Hora', fontsize=14)
        plt.xlabel('Hora del Día')
        plt.ylabel('Día de la Semana')
        if guardar_directorio:
            plt.savefig(ruta_dir / "P2_heatmap_desbalance.png", bbox_inches='tight')
        plt.show()

        # 3. Dispersión en Estaciones Top 10 Volátiles
        estaciones_criticas = df_plot.groupby('estacion_id')['b'].var().nlargest(10).index
        df_top = df_plot[df_plot['estacion_id'].isin(estaciones_criticas)]
        
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='estacion_id', y='b', data=df_top, order=estaciones_criticas)
        plt.axhline(y=-self.u_val_, color='red', linestyle='--', alpha=0.6)
        plt.axhline(y=self.u_val_, color='orange', linestyle='--', alpha=0.6)
        plt.title('Dispersión del Balance en las 10 Estaciones más Volátiles', fontsize=14)
        plt.xlabel('ID de Estación')
        plt.ylabel('Balance Normalizado (b)')
        if guardar_directorio:
            plt.savefig(ruta_dir / "P2_dispersion_estaciones_volatiles.png", bbox_inches='tight')
        plt.show()


# ==============================================================================
# EJECUCIÓN CLI
# ==============================================================================
@app.command()
def main(
    viajes_path: Path = INTERIM_DATA_DIR / "viajes_unificados.parquet",
    nomenclatura_path: Path = RAW_DATA_DIR / "nomenclatura_2026_09.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features_P2.parquet",
    figures_path: Path = Path("references/figures"),
    generar_graficas: bool = typer.Option(False, help="Activar para generar y visualizar gráficas de EDA."),
):
    logger.info("Iniciando la generación de características para el Problema 2...")
    
    df_viajes = pd.read_parquet(viajes_path)
    df_nomenclatura = pd.read_csv(nomenclatura_path)
    if 'id' in df_nomenclatura.columns:
        df_nomenclatura.rename(columns={'id': 'estacion_id', 'dpcapacity': 'capacidad'}, inplace=True)
        
    generador = FeatureEngineeringP2(ventana_prediccion=2, percentil_umbral=0.90)
    df_features = generador.fit(df_viajes, df_nomenclatura).transform(df_viajes, df_nomenclatura)
    
    if generar_graficas:
        logger.info(f"Generando gráficas de Análisis Exploratorio en: {figures_path}")
        generador.graficar_eda(df_features, guardar_directorio=figures_path)
    
    logger.info(f"Guardando características en: {output_path}")
    df_features.to_parquet(output_path, index=False)
    logger.success("Generación de características completada.")

if __name__ == "__main__":
    app()
