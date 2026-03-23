"""
Análisis de Rendimiento - Métricas de Performance para Inversiones de Congresistas

Calcula métricas de rendimiento por congresista, partido, comisión, y tipo de activo.
Compara retornos vs benchmarks (S&P 500).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import os
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calcular_metricas_congresista(
    df: pd.DataFrame,
    nombre: str = None
) -> Dict:
    """
    Calcula métricas de rendimiento para un congresista específico.

    Args:
        df: DataFrame con transacciones enriquecidas (con retornos)
        nombre: Nombre del congresista (opcional, si None calcula para todos)

    Returns:
        Dict con métricas:
        - total_operaciones: número total de operaciones
        - retorno_promedio: retorno promedio por operación
        - retorno_mediano: retorno mediano
        - retorno_total: retorno acumulado ponderado por monto
        - win_rate: % de operaciones positivas
        - sharpe_ratio: medida de riesgo-adjusted return (si hay suficientes datos)
        - alpha_promedio: outperformance vs S&P 500
    """
    if nombre:
        df = df[df["Name"].str.strip().str.lower() == nombre.lower().strip()]

    if len(df) == 0:
        return {
            "nombre": nombre or "todos",
            "total_operaciones": 0,
            "retorno_promedio": np.nan,
            "retorno_mediano": np.nan,
            "retorno_total": np.nan,
            "win_rate": np.nan,
            "sharpe_ratio": np.nan,
            "alpha_promedio": np.nan,
        }

    # Filtrar solo operaciones con retorno calculado
    df_validos = df[df["retorno_porcentual"].notna()]

    if len(df_validos) == 0:
        return {
            "nombre": nombre or "todos",
            "total_operaciones": len(df),
            "retorno_promedio": np.nan,
            "retorno_mediano": np.nan,
            "retorno_total": np.nan,
            "win_rate": np.nan,
            "sharpe_ratio": np.nan,
            "alpha_promedio": np.nan,
        }

    # Retorno promedio
    retorno_promedio = df_validos["retorno_porcentual"].mean()

    # Retorno mediano
    retorno_mediano = df_validos["retorno_porcentual"].median()

    # Win rate
    win_rate = (df_validos["retorno_porcentual"] > 0).mean()

    # Alpha promedio (vs S&P 500)
    alpha_promedio = np.nan
    if "alpha" in df_validos.columns and df_validos["alpha"].notna().any():
        alpha_promedio = df_validos["alpha"].mean()

    # Sharpe ratio simplificado (retorno / std, asumiendo risk-free = 0)
    sharpe_ratio = np.nan
    if len(df_validos) > 10 and df_validos["retorno_porcentual"].std() > 0:
        sharpe_ratio = retorno_promedio / df_validos["retorno_porcentual"].std()

    # Retorno total ponderado por monto (si está disponible)
    retorno_total = np.nan
    if "monto" in df_validos.columns:
        # Convertir monto a numérico si es categórico
        monto_num = convertir_monto_a_numerico(df_validos["monto"])
        if monto_num.notna().any():
            retorno_total = (df_validos["retorno_porcentual"] * monto_num).sum()

    return {
        "nombre": nombre or "todos",
        "total_operaciones": len(df_validos),
        "retorno_promedio": retorno_promedio,
        "retorno_mediano": retorno_mediano,
        "retorno_total": retorno_total,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe_ratio,
        "alpha_promedio": alpha_promedio,
    }


def convertir_monto_a_numerico(
    serie_montos: pd.Series
) -> pd.Series:
    """
    Convierte montos en formato categórico ($1001, $15001, etc.) a numérico.

    Los reportes del Congreso usan rangos:
    - $1,001 - $15,000 → codificado como "1001"
    - $15,001 - $50,000 → codificado como "15001"
    - $50,001 - $100,000 → codificado como "50001"
    - $100,001 - $250,000 → codificado como "100001"
    - $250,001 - $500,000 → codificado como "250001"
    - $500,001 - $1,000,000 → codificado como "500001"
    - > $1,000,000 → codificado como "1000001"

    Usamos el midpoint del rango como estimación.
    """
    rangos = {
        "1001": 8000,      # midpoint de 1001-15000
        "15001": 32500,    # midpoint de 15001-50000
        "50001": 75000,    # midpoint de 50001-100000
        "100001": 175000,  # midpoint de 100001-250000
        "250001": 375000,  # midpoint de 250001-500000
        "500001": 750000,  # midpoint de 500001-1000000
        "1000001": 1500000, # midpoint estimado para >1M
    }

    def mapear_monto(valor):
        if pd.isna(valor):
            return np.nan
        valor_str = str(valor).strip().replace(",", "").replace("$", "")
        for key, midpoint in rangos.items():
            if valor_str == key or valor_str.startswith(key):
                return midpoint
        # Intentar parsear como número directo
        try:
            return float(valor_str)
        except ValueError:
            return np.nan

    return serie_montos.apply(mapear_monto)


def calcular_metricas_por_grupo(
    df: pd.DataFrame,
    columna_grupo: str,
    min_operaciones: int = 10
) -> pd.DataFrame:
    """
    Calcula métricas de rendimiento agrupadas por una columna.

    Args:
        df: DataFrame con transacciones enriquecidas
        columna_grupo: Columna para agrupar (ej: "partido", "Owner")
        min_operaciones: Mínimo de operaciones para incluir grupo

    Returns:
        DataFrame con métricas por grupo
    """
    resultados = []

    grupos = df[columna_grupo].dropna().unique()

    for grupo in grupos:
        df_grupo = df[df[columna_grupo] == grupo]
        metricas = calcular_metricas_congresista(df_grupo)

        if metricas["total_operaciones"] >= min_operaciones:
            resultados.append({
                "grupo": grupo,
                **metricas
            })

    return pd.DataFrame(resultados)


def calcular_metricas_por_partido(
    df: pd.DataFrame,
    df_info: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Calcula métricas de rendimiento por partido político.

    Args:
        df: DataFrame con transacciones enriquecidas
        df_info: DataFrame con información de congresistas (para inferir partido)

    Returns:
        DataFrame con métricas por partido (D, R, I)
    """
    # Si tenemos info de congresistas, usar esa para partido
    if df_info is not None and "partido" in df_info.columns:
        # Merge para añadir partido a transacciones
        df_merged = df.merge(
            df_info[["name", "partido"]],
            left_on=df["Name"].str.strip().str.lower(),
            right_on=df_info["name"].str.strip().str.lower(),
            how="left"
        )
        df_merged["partido"] = df_merged["partido"].fillna("U")
        return calcular_metricas_por_grupo(df_merged, "partido")

    # Fallback: intentar inferir partido de los datos
    logger.warning("Sin información de partido, usando fallback...")
    return pd.DataFrame({
        "grupo": ["D", "R", "U"],
        "retorno_promedio": [np.nan, np.nan, np.nan],
        "nota": "Se requiere df_info para cálculo preciso"
    })


def calcular_metricas_por_comision(
    df: pd.DataFrame,
    df_info: pd.DataFrame
) -> pd.DataFrame:
    """
    Calcula métricas de rendimiento por tipo de comisión.

    Args:
        df: DataFrame con transacciones enriquecidas
        df_info: DataFrame con información de comisiones

    Returns:
        DataFrame con métricas por sector de comisión
    """
    # Unir transacciones con información de comisiones
    # Agrupar por sector de comisión

    sectores = ["banking", "energy", "health", "judiciary", "foreign", "other"]
    resultados = []

    for sector in sectores:
        # Encontrar congresistas en este sector
        congresistas_sector = df_info[df_info[f"comision_{sector}"] == 1]["name"].tolist()

        df_sector = df[df["Name"].str.strip().str.lower().isin(
            [n.lower() for n in congresistas_sector]
        )]

        metricas = calcular_metricas_congresista(df_sector)

        if metricas["total_operaciones"] >= 5:  # Mínimo 5 operaciones
            resultados.append({
                "sector": sector,
                **metricas
            })

    return pd.DataFrame(resultados)


def calcular_metricas_por_cargo(
    df: pd.DataFrame,
    df_info: pd.DataFrame
) -> pd.DataFrame:
    """
    Calcula métricas de rendimiento por tipo de cargo.

    Args:
        df: DataFrame con transacciones enriquecidas
        df_info: DataFrame con información de cargos (es_chair, es_ranking, etc.)

    Returns:
        DataFrame con métricas por tipo de cargo
    """
    resultados = []

    # Chairs
    chairs = df_info[df_info["es_chair"] == 1]["name"].tolist()
    df_chairs = df[df["Name"].str.strip().str.lower().isin([n.lower() for n in chairs])]
    metricas_chairs = calcular_metricas_congresista(df_chairs)
    metricas_chairs["tipo_cargo"] = "chair"
    resultados.append(metricas_chairs)

    # Ranking members
    ranking = df_info[df_info["es_ranking"] == 1]["name"].tolist()
    df_ranking = df[df["Name"].str.strip().str.lower().isin([n.lower() for n in ranking])]
    metricas_ranking = calcular_metricas_congresista(df_ranking)
    metricas_ranking["tipo_cargo"] = "ranking"
    resultados.append(metricas_ranking)

    # Members regulares (sin chair ni ranking)
    regular = df_info[
        (df_info["es_chair"] == 0) &
        (df_info["es_ranking"] == 0)
    ]["name"].tolist()
    df_regular = df[df["Name"].str.strip().str.lower().isin([n.lower() for n in regular])]
    metricas_regular = calcular_metricas_congresista(df_regular)
    metricas_regular["tipo_cargo"] = "member"
    resultados.append(metricas_regular)

    return pd.DataFrame(resultados)


def calcular_metricas_por_activo(
    df: pd.DataFrame,
    min_operaciones: int = 20
) -> pd.DataFrame:
    """
    Calcula métricas de rendimiento por activo (ticker).

    Args:
        df: DataFrame con transacciones enriquecidas
        min_operaciones: Mínimo de operaciones para incluir ticker

    Returns:
        DataFrame con métricas por ticker
    """
    resultados = []

    tickers = df["Ticker"].dropna().unique()

    for ticker in tickers:
        df_ticker = df[df["Ticker"] == ticker]
        metricas = calcular_metricas_congresista(df_ticker)

        if metricas["total_operaciones"] >= min_operaciones:
            resultados.append({
                "ticker": ticker,
                **metricas
            })

    df_resultado = pd.DataFrame(resultados)

    # Añadir información del activo desde yfinance si está disponible
    try:
        import yfinance as yf
        for idx, row in df_resultado.iterrows():
            ticker = row["ticker"]
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                df_resultado.loc[idx, "sector"] = info.get("sector", "N/A")
                df_resultado.loc[idx, "industry"] = info.get("industry", "N/A")
            except Exception:
                pass
    except Exception:
        pass

    return df_resultado


def calcular_metricas_por_tipo_operacion(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calcula métricas separadas para compras vs ventas.

    Args:
        df: DataFrame con transacciones enriquecidas

    Returns:
        DataFrame con métricas por tipo de operación
    """
    resultados = []

    # Compras
    df_compras = df[df["Type"].str.contains("Purchase", case=False, na=False)]
    metricas_compras = calcular_metricas_congresista(df_compras)
    metricas_compras["tipo"] = "compra"
    resultados.append(metricas_compras)

    # Ventas
    df_ventas = df[df["Type"].str.contains("Sale", case=False, na=False)]
    metricas_ventas = calcular_metricas_congresista(df_ventas)
    metricas_ventas["tipo"] = "venta"
    resultados.append(metricas_ventas)

    return pd.DataFrame(resultados)


def calcular_metricas_temporales(
    df: pd.DataFrame,
    periodo: str = "annual"
) -> pd.DataFrame:
    """
    Calcula métricas de rendimiento por periodo temporal.

    Args:
        df: DataFrame con transacciones enriquecidas
        periodo: "annual" o "quarterly"

    Returns:
        DataFrame con métricas por periodo
    """
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["Transaction.Date"])

    if periodo == "annual":
        df["periodo"] = df["fecha"].dt.year
    elif periodo == "quarterly":
        df["periodo"] = df["fecha"].dt.to_period("Q").astype(str)
    else:
        df["periodo"] = df["fecha"].dt.to_period(periodo).astype(str)

    resultados = []

    for periodo_val in sorted(df["periodo"].dropna().unique()):
        df_periodo = df[df["periodo"] == periodo_val]
        metricas = calcular_metricas_congresista(df_periodo)
        metricas["periodo"] = periodo_val
        resultados.append(metricas)

    return pd.DataFrame(resultados)


def generar_reporte_rendimiento_completo(
    df_transacciones: pd.DataFrame,
    df_info: pd.DataFrame,
    ruta_output: str = "datos/processed/reporte_rendimiento.csv"
) -> Dict[str, pd.DataFrame]:
    """
    Genera reporte completo de rendimiento con múltiples desagregaciones.

    Args:
        df_transacciones: DataFrame con transacciones enriquecidas (con retornos)
        df_info: DataFrame con información política de congresistas
        ruta_output: Ruta base para guardar reportes

    Returns:
        Dict de DataFrames con diferentes vistas del rendimiento
    """
    logger.info("=" * 60)
    logger.info("GENERANDO REPORTE DE RENDIMIENTO")
    logger.info("=" * 60)

    reportes = {}

    # 1. Métricas generales
    logger.info("\n[1/7] Métricas generales...")
    metricas_generales = calcular_metricas_congresista(df_transacciones)
    reportes["general"] = pd.DataFrame([metricas_generales])

    # 2. Por partido
    logger.info("[2/7] Por partido político...")
    reportes["partido"] = calcular_metricas_por_partido(df_transacciones, df_info)

    # 3. Por tipo de cargo
    logger.info("[3/7] Por tipo de cargo...")
    reportes["cargo"] = calcular_metricas_por_cargo(df_transacciones, df_info)

    # 4. Por sector de comisión
    logger.info("[4/7] Por sector de comisión...")
    reportes["sector"] = calcular_metricas_por_comision(df_transacciones, df_info)

    # 5. Por tipo de operación
    logger.info("[5/7] Por tipo de operación...")
    reportes["tipo_operacion"] = calcular_metricas_por_tipo_operacion(df_transacciones)

    # 6. Por activo (top 20)
    logger.info("[6/7] Por activo (top 20)...")
    reportes["activo"] = calcular_metricas_por_activo(df_transacciones, min_operaciones=5)

    # 7. Temporal (anual)
    logger.info("[7/7] Por año...")
    reportes["temporal"] = calcular_metricas_temporales(df_transacciones, "annual")

    # Guardar reportes
    os.makedirs(os.path.dirname(ruta_output), exist_ok=True)

    for nombre, df_reporte in reportes.items():
        ruta = f"{ruta_output.rsplit('.', 1)[0]}_{nombre}.csv"
        df_reporte.to_csv(ruta, index=False)
        logger.info(f"  → Guardado: {ruta}")

    # Imprimir resumen
    logger.info("\n" + "=" * 60)
    logger.info("RESUMEN DE RENDIMIENTO")
    logger.info("=" * 60)

    logger.info(f"\nRetorno promedio general: {metricas_generales['retorno_promedio']*100:.2f}%")
    logger.info(f"Win rate general: {metricas_generales['win_rate']*100:.1f}%")

    if len(reportes["partido"]) > 0:
        logger.info("\nPor partido:")
        for _, row in reportes["partido"].iterrows():
            logger.info(f"  {row['grupo']}: {row['retorno_promedio']*100:.2f}% ({row['total_operaciones']} ops)")

    if len(reportes["cargo"]) > 0:
        logger.info("\nPor cargo:")
        for _, row in reportes["cargo"].iterrows():
            logger.info(f"  {row['tipo_cargo']}: {row['retorno_promedio']*100:.2f}% ({row['total_operaciones']} ops)")

    return reportes


if __name__ == "__main__":
    # Ejecutar pipeline de análisis de rendimiento
    from enriquecer_precios import generar_dataset_enriquecido
    from comisiones_congreso import generar_dataset_congresistas

    # Cargar datos enriquecidos
    df_transacciones = generar_dataset_enriquecido()

    # Generar información de congresistas
    df_info = generar_dataset_congresistas(df_transacciones)

    # Generar reporte completo
    reportes = generar_reporte_rendimiento_completo(df_transacciones, df_info)

    print("\n" + "=" * 60)
    print("RENDERIMIENTO POR PARTIDO")
    print("=" * 60)
    print(reportes["partido"])

    print("\n" + "=" * 60)
    print("RENDERIMIENTO POR CARGO")
    print("=" * 60)
    print(reportes["cargo"])
