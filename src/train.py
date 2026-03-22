"""
Detector de Trading Sospechoso - Entrenamiento del Modelo

Entrena un modelo Random Forest con los datos históricos.
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Importar funciones de preprocessing
from preprocessing import (
    cargar_datos, limpiar_datos, crear_features,
    codificar_categoricas, preparar_features, dividir_train_test
)


def entrenar_modelo(
    ruta_datos: str = "datos/processed/datos_limpios.csv",
    ruta_modelo: str = "modelos/modelo_trading.pkl",
    random_state: int = 42
) -> dict:
    """
    Entrena el modelo de detección de trading sospechoso.

    Returns:
        Diccionario con métricas del modelo.
    """
    print("=" * 60)
    print("ENTRENAMIENTO DEL MODELO - DETECTOR DE INSIDER TRADING")
    print("=" * 60)

    # 1. Cargar datos
    print("\n[1/6] Cargando datos...")
    df = cargar_datos(ruta_datos)
    print(f"    → {len(df)} registros cargados")

    # 2. Limpiar datos
    print("\n[2/6] Limpiando datos...")
    df = limpiar_datos(df)
    print(f"    → {len(df)} registros después de limpieza")

    # 3. Feature engineering
    print("\n[3/6] Creando features...")
    df = crear_features(df)

    # 4. Codificar variables categóricas
    print("\n[4/6] Codificando variables categóricas...")
    columnas_categoricas = ["tipo", "cargo", "partido", "camara"]
    df, _ = codificar_categoricas(df, columnas_categoricas)

    # 5. Preparar X e y
    print("\n[5/6] Preparando features y target...")

    # Features que usará el modelo (deben coincidir con predecir.py)
    columnas_features = [
        "monto", "tipo_compra", "cargo_member", "partido_R",
        "camara_senado", "dias_hasta_evento", "return_anormal",
        "volumen_operaciones", "monto_log", "es_operacion_grande",
        "return_anormal_abs", "proximidad_evento"
    ]

    # Filtrar solo las columnas que existen
    columnas_features = [c for c in columnas_features if c in df.columns]

    X, y = preparar_features(df, columnas_features)
    print(f"    → {len(columnas_features)} features, {len(X)} muestras")
    print(f"    → Distribución: {sum(y)} sospechosas ({sum(y)/len(y)*100:.1f}%)")

    # 6. Dividir datos
    print("\n[6/6] Entrenando modelo...")
    X_train, X_test, y_train, y_test = dividir_train_test(
        X, y, test_size=0.2, random_state=random_state
    )

    # Entrenar Random Forest
    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",  # Manejar desbalance
        random_state=random_state,
        n_jobs=-1
    )

    modelo.fit(X_train, y_train)

    # Evaluar con cross-validation
    cv_scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"    → ROC-AUC (CV 5-fold): {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

    # Evaluar en test set
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 60)
    print("RESULTADOS EN CONJUNTO DE PRUEBA")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["Normal", "Sospechoso"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

    # Guardar modelo
    modelo_info = {
        "modelo": modelo,
        "features": columnas_features,
        "metricas": {
            "cv_roc_auc_mean": cv_scores.mean(),
            "cv_roc_auc_std": cv_scores.std(),
            "test_roc_auc": roc_auc_score(y_test, y_proba)
        }
    }

    joblib.dump(modelo_info, ruta_modelo)
    print(f"\n✓ Modelo guardado en: {ruta_modelo}")

    # Importancia de features
    print("\n" + "=" * 60)
    print("TOP 10 FEATURES MÁS IMPORTANTES")
    print("=" * 60)
    importances = modelo.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i, idx in enumerate(indices[:10]):
        print(f"    {i+1}. {columnas_features[idx]}: {importances[idx]:.3f}")

    return modelo_info


if __name__ == "__main__":
    # Generar datos sintéticos para demostración (REEMPLAZAR CON DATOS REALES)
    print("\n⚠️  Generando datos sintéticos de demostración...")
    print("    (Reemplazar con datos reales para producción)\n")

    np.random.seed(42)
    n = 1000

    datos_sinteticos = pd.DataFrame({
        "monto": np.random.lognormal(10, 2, n),  # Monto log-normal
        "tipo": np.random.choice(["compra", "venta"], n),
        "cargo": np.random.choice(["member", "chair", "leader", "whip"], n, p=[0.6, 0.25, 0.1, 0.05]),
        "partido": np.random.choice(["D", "R"], n),
        "camara": np.random.choice(["senado", "camara"], n),
        "dias_hasta_evento": np.random.exponential(30, n).astype(int),
        "return_anormal": np.random.normal(0, 10, n),
        "volumen_operaciones": np.random.poisson(10, n),
        "sospechoso": (np.random.random(n) < 0.15).astype(int)  # 15% sospechosas
    })

    # Guardar datos sintéticos
    datos_sinteticos.to_csv("datos/processed/datos_limpios.csv", index=False)

    # Entrenar
    entrenar_modelo()
