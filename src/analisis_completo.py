"""
Análisis Completo - Pipeline End-to-End para Trading de Congresistas

Orquestra todo el pipeline:
1. Carga de datos de transacciones
2. Enriquecimiento con precios de bolsa y retornos
3. Obtención de información política (comisiones, cargos, partidos)
4. Cálculo de métricas de rendimiento
5. Tests estadísticos de correlación
6. Generación de reportes y visualizaciones

Usage:
    python src/analisis_completo.py
"""

import pandas as pd
import numpy as np
import logging
import os
import json
from datetime import datetime
from pathlib import Path

# Importar módulos del proyecto
from enriquecer_precios import (
    generar_dataset_enriquecido,
    enriquecer_transacciones_con_retornos,
)
from comisiones_congreso import (
    generar_dataset_congresistas,
    construir_mapa_comisiones_congresistas,
)
from analisis_rendimiento import (
    generar_reporte_rendimiento_completo,
    calcular_metricas_congresista,
)
from analisis_correlacion import (
    generar_reporte_correlacion_completo,
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/analisis_completo.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Crear directorios
os.makedirs("logs", exist_ok=True)
os.makedirs("datos/raw", exist_ok=True)
os.makedirs("datos/processed", exist_ok=True)
os.makedirs("datos/cache", exist_ok=True)
os.makedirs("reportes", exist_ok=True)


def ejecutar_pipeline_completo(
    ruta_datos_input: str = "datos/raw/SenatorCleaned.csv",
    ruta_datos_enriquecidos: str = "datos/processed/transacciones_con_retornos.csv",
    ruta_info_congresistas: str = "datos/processed/congresistas_info.csv",
    ruta_reporte_rendimiento: str = "datos/processed/reporte_rendimiento.csv",
    ruta_reporte_correlacion: str = "datos/processed/reporte_correlacion.json",
    skip_enriquecer: bool = False,
    skip_comisiones: bool = False,
) -> dict:
    """
    Ejecuta el pipeline completo de análisis.

    Args:
        ruta_datos_input: Ruta al CSV original de transacciones
        ruta_datos_enriquecidos: Ruta para datos con retornos
        ruta_info_congresistas: Ruta para información política
        ruta_reporte_rendimiento: Ruta base para reportes de rendimiento
        ruta_reporte_correlacion: Ruta para reporte JSON de correlación
        skip_enriquecer: Si True, salta descarga de precios (usa datos existentes)
        skip_comisiones: Si True, salta obtención de comisiones

    Returns:
        Dict con DataFrames y resultados del análisis
    """
    logger.info("=" * 80)
    logger.info("PIPELINE DE ANÁLISIS DE TRADING DE CONGRESISTAS")
    logger.info("=" * 80)
    logger.info(f"Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    resultados = {}

    # =========================================================================
    # FASE 1: Enriquecimiento con precios de bolsa
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("FASE 1: ENRIQUECIMIENTO CON PRECIOS DE BOLSA")
    logger.info("=" * 80)

    if skip_enriquecer and os.path.exists(ruta_datos_enriquecidos):
        logger.info(f"Saltando enriquecimiento, usando datos existentes: {ruta_datos_enriquecidos}")
        df_transacciones = pd.read_csv(ruta_datos_enriquecidos)
    else:
        logger.info(f"Cargando datos desde: {ruta_datos_input}")
        df_transacciones = pd.read_csv(ruta_datos_input)
        logger.info(f"  → {len(df_transacciones)} transacciones cargadas")

        # Enriquecer con retornos
        df_transacciones = generar_dataset_enriquecido(
            ruta_input=ruta_datos_input,
            ruta_output=ruta_datos_enriquecidos,
            mostrar_progreso=True
        )

    resultados["df_transacciones"] = df_transacciones
    logger.info(f"✓ Fase 1 completa: {len(df_transacciones)} transacciones")

    # =========================================================================
    # FASE 2: Información política de congresistas
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("FASE 2: INFORMACIÓN POLÍTICA DE CONGRESISTAS")
    logger.info("=" * 80)

    if skip_comisiones and os.path.exists(ruta_info_congresistas):
        logger.info(f"Saltando comisiones, usando datos existentes: {ruta_info_congresistas}")
        df_info = pd.read_csv(ruta_info_congresistas)
    else:
        df_info = generar_dataset_congresistas(
            df_transacciones,
            ruta_output=ruta_info_congresistas
        )

    resultados["df_info"] = df_info
    logger.info(f"✓ Fase 2 completa: {len(df_info)} congresistas")

    # =========================================================================
    # FASE 3: Métricas de rendimiento
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("FASE 3: MÉTRICAS DE RENDIMIENTO")
    logger.info("=" * 80)

    reportes_rendimiento = generar_reporte_rendimiento_completo(
        df_transacciones,
        df_info,
        ruta_output=ruta_reporte_rendimiento
    )

    resultados["reportes_rendimiento"] = reportes_rendimiento
    logger.info(f"✓ Fase 3 completa: {len(reportes_rendimiento)} reportes generados")

    # =========================================================================
    # FASE 4: Tests estadísticos de correlación
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("FASE 4: ANÁLISIS DE CORRELACIÓN ESTADÍSTICA")
    logger.info("=" * 80)

    resultados_correlacion = generar_reporte_correlacion_completo(
        df_transacciones,
        df_info,
        ruta_output=ruta_reporte_correlacion
    )

    resultados["correlacion"] = resultados_correlacion
    logger.info(f"✓ Fase 4 completa: {len(resultados_correlacion)} tests realizados")

    # =========================================================================
    # FASE 5: Resumen ejecutivo
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("RESUMEN EJECUTIVO")
    logger.info("=" * 80)

    resumen = generar_resumen_ejecutivo(resultados)
    resultados["resumen"] = resumen

    # Guardar resumen
    ruta_resumen = "reportes/resumen_ejecutivo.json"
    os.makedirs(os.path.dirname(ruta_resumen), exist_ok=True)

    # Serializar para JSON
    def serialize(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [serialize(i) for i in obj]
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        return obj

    with open(ruta_resumen, "w", encoding="utf-8") as f:
        json.dump(serialize(resumen), f, indent=2)

    logger.info(f"✓ Resumen guardado en: {ruta_resumen}")

    return resultados


def generar_resumen_ejecutivo(resultados: dict) -> dict:
    """
    Genera un resumen ejecutivo de los hallazgos principales.

    Args:
        resultados: Dict con DataFrames y resultados del análisis

    Returns:
        Dict con resumen ejecutivo
    """
    df_transacciones = resultados.get("df_transacciones", pd.DataFrame())
    df_info = resultados.get("df_info", pd.DataFrame())
    reportes_rendimiento = resultados.get("reportes_rendimiento", {})
    correlacion = resultados.get("correlacion", {})

    resumen = {
        "fecha_generacion": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "datos": {
            "total_transacciones": len(df_transacciones),
            "transacciones_con_retorno": int(df_transacciones["retorno_porcentual"].notna().sum()),
            "total_congresistas": len(df_info),
            "periodo": "",  # Se podría calcular de las fechas
        },
        "rendimiento_general": {},
        "rendimiento_por_partido": [],
        "rendimiento_por_cargo": [],
        "tests_estadisticos": {},
        "hallazgos_clave": [],
    }

    # Rendimiento general
    if "general" in reportes_rendimiento and len(reportes_rendimiento["general"]) > 0:
        general = reportes_rendimiento["general"].iloc[0]
        resumen["rendimiento_general"] = {
            "retorno_promedio": float(general.get("retorno_promedio", np.nan)),
            "retorno_mediano": float(general.get("retorno_mediano", np.nan)),
            "win_rate": float(general.get("win_rate", np.nan)),
            "alpha_promedio": float(general.get("alpha_promedio", np.nan)),
        }

    # Rendimiento por partido
    if "partido" in reportes_rendimiento:
        df_partido = reportes_rendimiento["partido"]
        resumen["rendimiento_por_partido"] = df_partido.to_dict('records')

    # Rendimiento por cargo
    if "cargo" in reportes_rendimiento:
        df_cargo = reportes_rendimiento["cargo"]
        resumen["rendimiento_por_cargo"] = df_cargo.to_dict('records')

    # Tests estadísticos
    resumen["tests_estadisticos"] = {
        "t_test_partido": correlacion.get("t_test_partido", {}),
        "t_test_chair": correlacion.get("t_test_chair", {}),
        "anova_sector": correlacion.get("anova_sector", {}),
        "chi_square": correlacion.get("chi_square", {}),
        "regresion": correlacion.get("regresion", {}),
    }

    # Hallazgos clave
    hallazgos = []

    # Hallazgo 1: Rendimiento general
    if resumen["rendimiento_general"].get("retorno_promedio", 0) != 0:
        retorno_pct = resumen["rendimiento_general"]["retorno_promedio"] * 100
        hallazgos.append(f"Retorno promedio: {retorno_pct:.2f}%")

    # Hallazgo 2: Diferencia por partido
    t_test = correlacion.get("t_test_partido", {})
    if t_test.get("significativo", False):
        hallazgos.append(f"Partido: {t_test.get('conclusion', '')}")

    # Hallazgo 3: Diferencia por cargo
    chair_test = correlacion.get("t_test_chair", {})
    if chair_test.get("significativo", False):
        hallazgos.append(f"Cargo: {chair_test.get('conclusion', '')}")

    # Hallazgo 4: Diferencia por sector
    anova = correlacion.get("anova_sector", {})
    if anova.get("significativo", False):
        hallazgos.append(f"Sector comisión: {anova.get('conclusion', '')}")

    # Hallazgo 5: Variables significativas en regresión
    regresion = correlacion.get("regresion", {})
    sig_vars = regresion.get("variables_significativas", [])
    if sig_vars:
        hallazgos.append(f"Regresión: variables significativas = {sig_vars}")

    resumen["hallazgos_clave"] = hallazgos

    return resumen


def imprimir_resumen(resultados: dict):
    """Imprime un resumen formateado en la consola."""
    resumen = resultados.get("resumen", {})

    print("\n" + "=" * 80)
    print("RESUMEN EJECUTIVO - TRADING DE CONGRESISTAS")
    print("=" * 80)

    print(f"\n📊 DATOS")
    print(f"   Transacciones: {resumen.get('datos', {}).get('total_transacciones', 'N/A')}")
    print(f"   Congresistas: {resumen.get('datos', {}).get('total_congresistas', 'N/A')}")

    print(f"\n📈 RENDIMIENTO GENERAL")
    rend = resumen.get("rendimiento_general", {})
    print(f"   Retorno promedio: {rend.get('retorno_promedio', np.nan)*100:.2f}%")
    print(f"   Win rate: {rend.get('win_rate', np.nan)*100:.1f}%")
    print(f"   Alpha vs S&P 500: {rend.get('alpha_promedio', np.nan)*100:.2f}%")

    print(f"\n🔴 RENDIMIENTO POR PARTIDO")
    for row in resumen.get("rendimiento_por_partido", []):
        print(f"   {row.get('grupo', 'N/A')}: {row.get('retorno_promedio', np.nan)*100:.2f}% ({row.get('total_operaciones', 0)} ops)")

    print(f"\n🎯 RENDIMIENTO POR CARGO")
    for row in resumen.get("rendimiento_por_cargo", []):
        print(f"   {row.get('tipo_cargo', 'N/A')}: {row.get('retorno_promedio', np.nan)*100:.2f}% ({row.get('total_operaciones', 0)} ops)")

    print(f"\n🧪 TESTS ESTADÍSTICOS")
    tests = resumen.get("tests_estadisticos", {})
    for test_name, resultado in tests.items():
        sig = "✓ SIGNIFICATIVO" if resultado.get("significativo", False) else "○ no significativo"
        print(f"   {test_name}: {sig}")

    print(f"\n💡 HALLAZGOS CLAVE")
    for hallazgo in resumen.get("hallazgos_clave", []):
        print(f"   • {hallazgo}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Ejecutar pipeline completo
    resultados = ejecutar_pipeline_completo()

    # Imprimir resumen
    imprimir_resumen(resultados)

    print("\n✓ Pipeline completo ejecutado exitosamente!")
    print(f"  - Datos enriquecidos: datos/processed/transacciones_con_retornos.csv")
    print(f"  - Info congresistas: datos/processed/congresistas_info.csv")
    print(f"  - Reportes rendimiento: datos/processed/reporte_rendimiento_*.csv")
    print(f"  - Reporte correlación: datos/processed/reporte_correlacion.json")
    print(f"  - Resumen ejecutivo: reportes/resumen_ejecutivo.json")
