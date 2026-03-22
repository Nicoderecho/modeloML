"""
Detector de Trading Sospechoso - Preprocesamiento de Datos

Limpia y transforma los datos crudos para el modelo.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


def cargar_datos(ruta: str) -> pd.DataFrame:
    """Carga datos desde CSV."""
    return pd.read_csv(ruta)


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia valores faltantes y outliers."""
    df = df.copy()

    # Eliminar filas con valores faltantes críticos
    df = df.dropna(subset=["monto", "tipo", "sospechoso"])

    # Rellenar valores faltantes en features opcionales
    df["cargo"] = df["cargo"].fillna("member")
    df["partido"] = df["partido"].fillna("I")  # Independiente por defecto
    df["return_anormal"] = df["return_anormal"].fillna(0)

    # Eliminar montos extremos (posibles errores)
    if "monto" in df.columns:
        q99 = df["monto"].quantile(0.99)
        df = df[df["monto"] <= q99]

    return df


def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering: crea nuevas variables."""
    df = df.copy()

    # Monto relativo (log)
    df["monto_log"] = np.log1p(df["monto"])

    # Indicador de operación grande
    df["es_operacion_grande"] = (df["monto"] > df["monto"].median()).astype(int)

    # Return anormal absoluto
    df["return_anormal_abs"] = df["return_anormal"].abs()

    # Proximidad a evento legislativo (categoría)
    df["proximidad_evento"] = pd.cut(
        df["dias_hasta_evento"],
        bins=[-np.inf, 7, 30, 90, np.inf],
        labels=[3, 2, 1, 0]  # 3=muy cerca, 0=lejos
    ).astype(int)

    return df


def codificar_categoricas(
    df: pd.DataFrame,
    columnas: List[str],
    encoder=None
) -> Tuple[pd.DataFrame, object]:
    """
    Codifica variables categóricas usando One-Hot Encoding.

    Returns:
        DataFrame codificado y el encoder para usar en producción.
    """
    df = df.copy()

    if encoder is None:
        # Crear dummies
        df = pd.get_dummies(df, columns=columnas, drop_first=True)
        return df, None

    # Usar encoder existente (para datos nuevos)
    for col in columnas:
        if col in df.columns:
            df[col] = encoder[col].transform(df[col])

    return df, encoder


def preparar_features(
    df: pd.DataFrame,
    columnas_features: List[str],
    columna_target: str = "sospechoso"
) -> Tuple[np.ndarray, np.ndarray]:
    """Separa X (features) e y (target)."""
    X = df[columnas_features].values
    y = df[columna_target].values
    return X, y


def dividir_train_test(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Divide en conjuntos de entrenamiento y prueba."""
    from sklearn.model_selection import train_test_split
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
