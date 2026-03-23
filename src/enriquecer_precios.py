"""
Enriquecer Precios - Datos de Bolsa para Transacciones de Congresistas

Descarga precios históricos desde Yahoo Finance y calcula retornos
para cada transacción de congresistas.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Tuple
import logging
import os
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache directory para datos de precios
CACHE_DIR = "datos/cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def obtener_precios_yfinance(
    ticker: str,
    start_date: str,
    end_date: str,
    usar_cache: bool = True
) -> Optional[pd.DataFrame]:
    """
    Descarga precios históricos desde Yahoo Finance.

    Args:
        ticker: Símbolo del activo (ej: "AAPL", "MSFT")
        start_date: Fecha de inicio (YYYY-MM-DD)
        end_date: Fecha de fin (YYYY-MM-DD)
        usar_cache: Si True, usa datos cacheados si existen

    Returns:
        DataFrame con columnas: Date, Open, High, Low, Close, Volume
        o None si no se pudieron obtener datos
    """
    # Normalizar ticker
    ticker = ticker.strip().upper()

    # Tickers especiales que yfinance no maneja bien
    tickers_problematicos = {
        "--": None,  # No ticker
        "GENERAL ELECTRIC": "GE",
        "WFC": "WFC",
    }

    if ticker in ["--", "-", ""] or "BOND" in ticker or "CORPORATE" in ticker:
        logger.debug(f"    Ticker inválido: {ticker}")
        return None

    # Intentar usar cache
    cache_file = Path(CACHE_DIR) / f"{ticker}_prices.csv"
    if usar_cache and cache_file.exists():
        try:
            df_cache = pd.read_csv(cache_file)
            df_cache["Date"] = pd.to_datetime(df_cache["Date"])
            if df_cache["Date"].min() <= pd.to_datetime(start_date) and \
               df_cache["Date"].max() >= pd.to_datetime(end_date):
                logger.debug(f"    Usando cache para {ticker}")
                return df_cache
        except Exception:
            pass

    # Descargar desde yfinance
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)

        if df is None or df.empty:
            logger.debug(f"    No hay datos para {ticker}")
            return None

        # Reset index para tener Date como columna
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

        # Normalizar nombres de columnas
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })

        # Guardar en cache
        if usar_cache:
            df.to_csv(cache_file, index=False)

        return df[["Date", "close", "volume"]]

    except Exception as e:
        logger.debug(f"    Error descargando {ticker}: {e}")
        return None


def get_close_price(
    ticker: str,
    fecha: str,
    precios_df: Optional[pd.DataFrame] = None
) -> Optional[float]:
    """
    Obtiene el precio de cierre en una fecha específica.

    Args:
        ticker: Símbolo del activo
        fecha: Fecha de la transacción
        precios_df: DataFrame con precios históricos (opcional, para evitar re-descargar)

    Returns:
        Precio de cierre o None si no disponible
    """
    fecha_dt = pd.to_datetime(fecha)

    if precios_df is not None:
        # Buscar precio más cercano a la fecha
        mask = precios_df["Date"] == fecha_dt
        if mask.any():
            return precios_df.loc[mask, "close"].iloc[0]

        # Si no hay precio exacto, usar el más cercano (dentro de 5 días)
        diff = (precios_df["Date"] - fecha_dt).abs()
        if diff.min() <= timedelta(days=5):
            idx = diff.idxmin()
            return precios_df.loc[idx, "close"]

        return None

    # Descargar precios para esa fecha
    start = (fecha_dt - timedelta(days=10)).strftime("%Y-%m-%d")
    end = (fecha_dt + timedelta(days=10)).strftime("%Y-%m-%d")

    precios = obtener_precios_yfinance(ticker, start, end)
    if precios is None:
        return None

    return get_close_price(ticker, fecha, precios)


def calcular_retorno_compra(
    ticker: str,
    fecha_compra: str,
    dias_hold: int = 90,
    precios_df: Optional[pd.DataFrame] = None
) -> Optional[float]:
    """
    Calcula el retorno de una compra manteniendo por N días.

    Args:
        ticker: Símb del activo
        fecha_compra: Fecha de la compra
        dias_hold: Días a mantener el activo
        precios_df: DataFrame con precios históricos

    Returns:
        Retorno porcentual (ej: 0.15 = 15%) o None
    """
    fecha_dt = pd.to_datetime(fecha_compra)
    fecha_venta = fecha_dt + timedelta(days=dias_hold)

    # Obtener rango de fechas
    start = (fecha_dt - timedelta(days=10)).strftime("%Y-%m-%d")
    end = (fecha_venta + timedelta(days=10)).strftime("%Y-%m-%d")

    # Descargar precios si no provistos
    if precios_df is None:
        precios_df = obtener_precios_yfinance(ticker, start, end)

    if precios_df is None or precios_df.empty:
        return None

    # Precio de entrada (más cercano a fecha_compra)
    diff_compra = (precios_df["Date"] - fecha_dt).abs()
    if diff_compra.min() > timedelta(days=5):
        return None
    idx_compra = diff_compra.idxmin()
    precio_entrada = precios_df.loc[idx_compra, "close"]

    # Precio de salida (más cercano a fecha_venta, pero después de compra)
    df_venta = precios_df[precios_df["Date"] >= fecha_compra]
    if df_venta.empty:
        return None

    # Buscar precio en fecha_venta o el más cercano después
    diff_venta = (df_venta["Date"] - fecha_venta).abs()
    if diff_venta.min() > timedelta(days=10):
        # Si no hay precio en fecha_venta, usar último disponible
        precio_salida = df_venta["close"].iloc[-1]
    else:
        idx_venta = diff_venta.idxmin()
        precio_salida = precios_df.loc[idx_venta, "close"]

    # Calcular retorno
    if precio_entrada is None or precio_entrada <= 0 or precio_salida is None:
        return None

    retorno = (precio_salida - precio_entrada) / precio_entrada
    return retorno


def calcular_retorno_venta(
    ticker: str,
    fecha_venta: str,
    dias_lookback: int = 30,
    precios_df: Optional[pd.DataFrame] = None
) -> Optional[float]:
    """
    Calcula el retorno de una venta (asume compra dias_lookback días antes).

    Para ventas, no sabemos cuándo se compró originalmente.
    Asumimos que se compró N días antes para estimar el retorno.

    Args:
        ticker: Símbolo del activo
        fecha_venta: Fecha de la venta
        dias_lookback: Días asumidos de holding antes de la venta
        precios_df: DataFrame con precios históricos

    Returns:
        Retorno porcentual estimado o None
    """
    fecha_venta_dt = pd.to_datetime(fecha_venta)
    fecha_compra = fecha_venta_dt - timedelta(days=dias_lookback)

    # El retorno es desde la perspectiva de cuando se compró
    # Retorno = (precio_venta - precio_compra) / precio_compra
    retorno = calcular_retorno_compra(
        ticker,
        fecha_compra.strftime("%Y-%m-%d"),
        dias_hold=dias_lookback,
        precios_df=precios_df
    )

    return retorno


def calcular_retorno_anualizado(
    retorno: float,
    dias_hold: int
) -> Optional[float]:
    """
    Calcula el retorno anualizado a partir de un retorno y días de holding.

    Args:
        retorno: Retorno porcentual (ej: 0.15)
        dias_hold: Días mantenidos

    Returns:
        Retorno anualizado o None
    """
    if retorno is None or dias_hold <= 0:
        return None

    # Annualizar: (1 + retorno) ^ (365/dias) - 1
    try:
        retorno_anualizado = (1 + retorno) ** (365 / dias_hold) - 1
        return retorno_anualizado
    except (ValueError, ZeroDivisionError):
        return None


def obtener_retorno_sp500(
    fecha: str,
    dias_hold: int = 90
) -> Optional[float]:
    """
    Obtiene el retorno del S&P 500 en un periodo para benchmark.

    Args:
        fecha: Fecha de inicio
        dias_hold: Días de holding

    Returns:
        Retorno del S&P 500 o None
    """
    ticker_sp500 = "^GSPC"  # S&P 500
    retorno = calcular_retorno_compra(ticker_sp500, fecha, dias_hold)
    return retorno


def enriquecer_transacciones_con_retornos(
    df: pd.DataFrame,
    dias_hold_compra: int = 90,
    dias_lookback_venta: int = 30,
    mostrar_progreso: bool = True
) -> pd.DataFrame:
    """
    Enriquece un DataFrame de transacciones con retornos calculados.

    Args:
        df: DataFrame con columnas: Ticker, Type, Transaction.Date
        dias_hold_compra: Días a mantener para compras
        dias_lookback_venta: Días lookback para ventas
        mostrar_progreso: Si True, muestra log de progreso

    Returns:
        DataFrame con columnas añadidas:
        - retorno_porcentual: Retorno de la operación
        - retorno_anualizado: Retorno anualizado
        - retorno_sp500: Retorno del S&P 500 en mismo periodo
        - alpha: retorno_porcentual - retorno_sp500
    """
    df = df.copy()

    # Inicializar columnas
    df["retorno_porcentual"] = np.nan
    df["retorno_anualizado"] = np.nan
    df["retorno_sp500"] = np.nan
    df["alpha"] = np.nan
    df["dias_hold"] = np.nan

    # Cache de precios por ticker para evitar re-descargas
    cache_precios = {}

    total = len(df)

    for idx, row in df.iterrows():
        ticker = row.get("Ticker", "")
        tipo = row.get("Type", "")
        fecha = row.get("Transaction.Date", "")

        if mostrar_progreso and idx % 100 == 0:
            logger.info(f"  Procesando {idx}/{total} ({idx/total*100:.1f}%)")

        # Saltar si no hay ticker válido
        if not ticker or ticker in ["--", "-", ""] or "BOND" in ticker.upper():
            continue

        # Normalizar fecha
        try:
            fecha_dt = pd.to_datetime(fecha)
            fecha_str = fecha_dt.strftime("%Y-%m-%d")
        except Exception:
            continue

        # Obtecer o descargar precios para este ticker
        if ticker not in cache_precios:
            start = (fecha_dt - timedelta(days=365)).strftime("%Y-%m-%d")
            end = (fecha_dt + timedelta(days=365)).strftime("%Y-%m-%d")
            cache_precios[ticker] = obtener_precios_yfinance(ticker, start, end)

        precios_df = cache_precios.get(ticker)

        # Calcular retorno según tipo de operación
        retorno = None
        dias_hold = None

        if "Purchase" in tipo:
            retorno = calcular_retorno_compra(
                ticker, fecha_str, dias_hold_compra, precios_df
            )
            dias_hold = dias_hold_compra
        elif "Sale" in tipo:
            retorno = calcular_retorno_venta(
                ticker, fecha_str, dias_lookback_venta, precios_df
            )
            dias_hold = dias_lookback_venta

        if retorno is not None:
            df.loc[idx, "retorno_porcentual"] = retorno
            df.loc[idx, "retorno_anualizado"] = calcular_retorno_anualizado(retorno, dias_hold)
            df.loc[idx, "dias_hold"] = dias_hold

            # Benchmark vs S&P 500
            retorno_sp500 = obtener_retorno_sp500(fecha_str, dias_hold)
            if retorno_sp500 is not None:
                df.loc[idx, "retorno_sp500"] = retorno_sp500
                df.loc[idx, "alpha"] = retorno - retorno_sp500

    if mostrar_progreso:
        logger.info(f"  Procesamiento completo: {total} transacciones")
        completadas = df["retorno_porcentual"].notna().sum()
        logger.info(f"  Retornos calculados: {completadas}/{total} ({completadas/total*100:.1f}%)")

    return df


def generar_dataset_enriquecido(
    ruta_input: str = "datos/raw/SenatorCleaned.csv",
    ruta_output: str = "datos/processed/transacciones_con_retornos.csv",
    **kwargs
) -> pd.DataFrame:
    """
    Pipeline completo: carga datos, enriquece con retornos y guarda.

    Args:
        ruta_input: Ruta al CSV con transacciones
        ruta_output: Ruta para guardar datos enriquecidos
        **kwargs: Argumentos para enriquecer_transacciones_con_retornos

    Returns:
        DataFrame enriquecido
    """
    logger.info("=" * 60)
    logger.info("ENRIQUECIMIENTO DE DATOS CON RETORNOS DE BOLSA")
    logger.info("=" * 60)

    # Cargar datos
    logger.info(f"\n[1/3] Cargando datos desde {ruta_input}...")
    df = pd.read_csv(ruta_input)
    logger.info(f"    → {len(df)} transacciones cargadas")

    # Normalizar nombres de columnas
    df = df.rename(columns={
        "Transaction.Date": "Transaction.Date",
        "Type": "Type",
        "Ticker": "Ticker"
    })

    # Enriquecer con retornos
    logger.info("\n[2/3] Calculando retornos...")
    df_enriquecido = enriquecer_transacciones_con_retornos(df, **kwargs)

    # Guardar resultado
    logger.info("\n[3/3] Guardando datos enriquecidos...")
    os.makedirs(os.path.dirname(ruta_output), exist_ok=True)
    df_enriquecido.to_csv(ruta_output, index=False)
    logger.info(f"    → Datos guardados en: {ruta_output}")

    # Resumen estadístico
    logger.info("\n" + "=" * 60)
    logger.info("RESUMEN DE RETORNOS")
    logger.info("=" * 60)

    df_validos = df_enriquecido[df_enriquecido["retorno_porcentual"].notna()]

    if len(df_validos) > 0:
        logger.info(f"\nRetorno promedio: {df_validos['retorno_porcentual'].mean()*100:.2f}%")
        logger.info(f"Retorno mediano: {df_validos['retorno_porcentual'].median()*100:.2f}%")
        logger.info(f"Retorno std: {df_validos['retorno_porcentual'].std()*100:.2f}%")
        logger.info(f"Win rate: {(df_validos['retorno_porcentual'] > 0).sum()/len(df_validos)*100:.1f}%")

        # Comparación con S&P 500
        if df_validos["alpha"].notna().any():
            logger.info(f"\nAlpha promedio vs S&P 500: {df_validos['alpha'].mean()*100:.2f}%")

    return df_enriquecido


if __name__ == "__main__":
    # Ejecutar pipeline completo
    df_resultado = generar_dataset_enriquecido()

    print("\n" + "=" * 60)
    print("MUESTRAS DE DATOS ENRIQUECIDOS")
    print("=" * 60)
    print(df_resultado[["Name", "Ticker", "Type", "Transaction.Date",
                        "retorno_porcentual", "retorno_anualizado", "alpha"]].head(10))
