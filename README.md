# Detector de Trading Sospechoso - Congresistas USA

Sistema de Machine Learning para identificar operaciones financieras sospechosas (insider trading) de congresistas y senadores de EE.UU.

## Quick Start

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Entrenar el modelo (con datos sintéticos de demo)
```bash
cd src
python train.py
```

### 3. Hacer predicciones
```python
from predecir import predecir_trading_sospechoso

operacion = {
    "monto": 50000,
    "tipo": "compra",
    "cargo": "chair",
    "partido": "D",
    "camara": "senado",
    "dias_hasta_evento": 3,
    "return_anormal": 15.5,
    "volumen_operaciones": 20,
}

resultado = predecir_trading_sospechoso(operacion)
print(resultado)
# {'sospechoso': True, 'confianza': 0.87, 'nivel_seguridad': 'Alta', ...}
```

### 4. Aprender ML
```bash
jupyter notebook notebooks/01_analisis.ipynb
```

## Estructura
```
├── datos/           # Datos (raw y processed)
├── modelos/         # Modelos entrenados (.pkl)
├── notebooks/       # Jupyter notebooks educativos
├── src/
│   ├── predecir.py      # Función de predicción
│   ├── preprocessing.py # Limpieza de datos
│   └── train.py         # Entrenamiento
├── requirements.txt
└── README.md
```

## Conceptos Clave

- **Feature:** Variable de entrada (monto, tipo, cargo...)
- **Target:** Lo que predecimos (sospechoso: Sí/No)
- **Confianza:** `predict_proba()` devuelve probabilidades 0-1
  - Alta (≥80%): Predicción fiable
  - Media (50-80%): Revisar manualmente
  - Baja (<50%): Muy incierto

## Fuentes de Datos

- House.gov: [Periodic Transaction Reports](https://disclosures.clerk.house.gov/PublicDisclosure/AnnualReports.aspx)
- Senate.gov: Financial Disclosures
- Yahoo Finance (yfinance): Datos de mercado

## Warning

Este proyecto es educativo. No constituye asesoramiento financiero. Los resultados deben validarse manualmente.
