"""
Detector de Trading Sospechoso - Función de Predicción

Esta función recibe datos de una operación financiera y devuelve:
- Si es sospechosa de insider trading
- Nivel de confianza de la predicción
- Nivel de seguridad (Alta/Media/Baja)
"""

import joblib
import numpy as np
from typing import Dict, Any


def predecir_trading_sospechoso(
    info_operacion: Dict[str, Any],
    ruta_modelo: str = "modelos/modelo_trading.pkl"
) -> Dict[str, Any]:
    """
    Predice si una operación financiera es sospechosa de insider trading.

    Args:
        info_operacion: Diccionario con los datos de la operación.
            Requiere:
            - monto: float (valor de la operación en USD)
            - tipo: str ('compra' o 'venta')
            - cargo: str (posición del congresista)
            - partido: str ('D' o 'R')
            - camara: str ('senado' o 'camara')
            - dias_hasta_evento: int (días antes de evento legislativo)
            - return_anormal: float (% de return anormal vs mercado)
            - volumen_operaciones: int (número de operaciones recientes)

        ruta_modelo: Ruta al archivo del modelo entrenado (.pkl)

    Returns:
        Diccionario con:
            - sospechoso: bool
            - confianza: float (0.0 a 1.0)
            - nivel_seguridad: str ('Alta', 'Media', 'Baja')
            - probabilidades: dict {'no_sospechoso': float, 'sospechoso': float}
    """
    # Cargar modelo
    modelo = joblib.load(ruta_modelo)

    # Extraer features en el orden esperado por el modelo
    features = np.array([[
        info_operacion.get("monto", 0),
        1 if info_operacion.get("tipo") == "compra" else 0,
        _codificar_cargo(info_operacion.get("cargo", "")),
        1 if info_operacion.get("partido") == "D" else 0,
        1 if info_operacion.get("camara") == "senado" else 0,
        info_operacion.get("dias_hasta_evento", 0),
        info_operacion.get("return_anormal", 0),
        info_operacion.get("volumen_operaciones", 0),
    ]])

    # Predicción
    prediccion = modelo.predict(features)[0]
    probabilidades = modelo.predict_proba(features)[0]

    # Confianza = probabilidad de la clase predicha
    confianza = float(max(probabilidades))

    # Determinar nivel de seguridad
    if confianza >= 0.80:
        nivel_seguridad = "Alta"
    elif confianza >= 0.50:
        nivel_seguridad = "Media"
    else:
        nivel_seguridad = "Baja"

    return {
        "sospechoso": bool(prediccion == 1),
        "confianza": confianza,
        "nivel_seguridad": nivel_seguridad,
        "probabilidades": {
            "no_sospechoso": float(probabilidades[0]),
            "sospechoso": float(probabilidades[1]),
        }
    }


def _codificar_cargo(cargo: str) -> int:
    """Codifica el cargo del congresista a número."""
    cargos_jerarquia = {
        "leader": 5,
        "speaker": 4,
        "whip": 3,
        "chair": 2,
        "member": 1,
        "": 0,
    }
    cargo_lower = cargo.lower()
    for key, value in cargos_jerarquia.items():
        if key in cargo_lower:
            return value
    return 0


# ============================================================
# EJEMPLO DE USO (descomentar para probar)
# ============================================================
if __name__ == "__main__":
    # Ejemplo con datos simulados
    operacion_ejemplo = {
        "monto": 50000,
        "tipo": "compra",
        "cargo": "chair",  # Presidente de comité
        "partido": "D",
        "camara": "senado",
        "dias_hasta_evento": 3,  # 3 días antes de vote
        "return_anormal": 15.5,  # 15.5% por encima del mercado
        "volumen_operaciones": 20,
    }

    try:
        resultado = predecir_trading_sospechoso(operacion_ejemplo)
        print("=" * 50)
        print("RESULTADO DE PREDICCIÓN")
        print("=" * 50)
        print(f"Sospechoso: {resultado['sospechoso']}")
        print(f"Confianza: {resultado['confianza']:.2%}")
        print(f"Nivel de Seguridad: {resultado['nivel_seguridad']}")
        print(f"Probabilidades: {resultado['probabilidades']}")
    except FileNotFoundError:
        print("Modelo no encontrado. Ejecuta primero train.py para entrenarlo.")
