#!/bin/bash

# ============================================================
# Quick Start - Tareas comunes del proyecto
# ============================================================

# Cargar entorno virtual si existe
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

case "$1" in
    "")
        echo "Uso: ./run.sh [comando]"
        echo ""
        echo "Comandos disponibles:"
        echo "  setup         - Ejecutar configuración inicial"
        echo "  train         - Entrenar el modelo"
        echo "  predict       - Hacer una predicción de prueba"
        echo "  notebook      - Iniciar Jupyter Notebook"
        echo "  data          - Descargar datos"
        echo "  clean         - Limpiar archivos temporales"
        echo "  status        - Ver estado del proyecto"
        ;;

    setup)
        bash setup.sh
        ;;

    train)
        echo "Entrenando modelo..."
        cd src
        python train.py
        ;;

    predict)
        echo "Ejecutando predicción de prueba..."
        cd src
        python predecir.py
        ;;

    notebook)
        echo "Iniciando Jupyter Notebook..."
        jupyter notebook notebooks/
        ;;

    data)
        echo "Descargando datos..."
        cd src
        python obtener_datos.py
        ;;

    clean)
        echo "Limpiando..."
        find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find . -type f -name "*.pyc" -delete 2>/dev/null || true
        echo "✓ Limpieza completada"
        ;;

    status)
        echo "=============================================="
        echo "ESTADO DEL PROYECTO"
        echo "=============================================="
        echo ""
        echo "Estructura:"
        tree -L 2 -I '__pycache__|*.pyc|venv' 2>/dev/null || ls -R
        echo ""
        echo "Modelo entrenado:"
        if [ -f "modelos/modelo_trading.pkl" ]; then
            echo "  ✓ Modelo existe"
            ls -lh modelos/modelo_trading.pkl
        else
            echo "  ✗ Modelo NO existe (ejecuta: ./run.sh train)"
        fi
        echo ""
        echo "Datos disponibles:"
        if [ -f "datos/processed/datos_limpios.csv" ]; then
            echo "  ✓ Datos existen"
            wc -l datos/processed/datos_limpios.csv
        else
            echo "  ✗ Datos NO existen"
        fi
        ;;
esac
