# CLAUDE.md - Detector de Trading Sospechoso

## Proyecto

Sistema de Machine Learning para identificar operaciones financieras sospechosas (insider trading) de congresistas y senadores de EE.UU.

## Objetivo

Aprendizaje de ML + proyecto funcional. El modelo aprende de datos históricos para detectar patrones que sugieren trading con información privilegiada.

## Stack Tecnológico

- **Python 3** con entorno virtual (venv)
- **scikit-learn** - Modelo Random Forest
- **pandas/numpy** - Manipulación de datos
- **Jupyter Notebook** - Aprendizaje y exploración
- **yfinance** - Datos de mercado

## Estructura del Proyecto

```
modeloML/
├── src/
│   ├── predecir.py          # Función de predicción con confianza
│   ├── preprocessing.py     # Limpieza y transformación de datos
│   ├── train.py             # Entrenamiento del modelo
│   └── obtener_datos.py     # Descarga de datos públicos
├── datos/
│   ├── raw/                  # Datos originales
│   └── processed/            # Datos limpios para entrenar
├── modelos/
│   └── modelo_trading.pkl    # Modelo entrenado
├── notebooks/
│   └── 01_analisis.ipynb    # Notebook educativo de ML
├── setup.sh                  # Configuración inicial
├── run.sh                    # Comandos rápidos
├── requirements.txt          # Dependencias Python
└── README.md                 # Documentación
```

## Comandos Rápidos

```bash
./run.sh setup     # Configurar entorno (primera vez)
./run.sh train     # Entrenar modelo
./run.sh predict   # Probar predicción
./run.sh notebook  # Abrir Jupyter
./run.sh data      # Descargar datos
./run.sh status    # Ver estado del proyecto
```

## Conceptos Clave

- **Features (X)**: Variables de entrada - monto, tipo, cargo, partido, cámara, días hasta evento, return anormal
- **Target (y)**: Lo que predecimos - `sospechoso` (1) o `normal` (0)
- **Grado de seguridad**: `predict_proba()` devuelve confianza de 0.0 a 1.0
  - ≥80%: Alta seguridad
  - 50-80%: Media seguridad
  - <50%: Baja seguridad

## Fuentes de Datos

- [House.gov - Periodic Transaction Reports](https://disclosures.clerk.house.gov/PublicDisclosure/AnnualReports.aspx)
- [Senate.gov - Financial Disclosures](https://www.senate.gov/legal/disclosure.htm)
- [unusual_whales GitHub](https://github.com/unusual_whales/insider-trading-data) - datos agregados

## Modelo Actual

- **Algoritmo**: Random Forest Classifier
- **Problema**: Clasificación binaria
- **Métricas objetivo**: ROC-AUC, Precision, Recall

## Próximos Pasos (Backlog)

1. [ ] Obtener datos reales de fuentes oficiales
2. [ ] Feature engineering: crear más variables útiles
3. [ ] Evaluar modelo con datos de casos conocidos de insider trading
4. [ ] Implementar modelo XGBoost para comparar
5. [ ] Crear API REST para servir predicciones
6. [ ] Dashboard web simple para visualizar resultados

## Conceptos de ML a Aprender

- Train/Test Split y Cross-Validation
- Overfitting vs Underfitting
- Feature Importance
- Matriz de confusión (Precision, Recall, F1)
- ROC-AUC
