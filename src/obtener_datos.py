"""
Obtener datos de transacciones de congresistas desde fuentes oficiales.

- House.gov: Periodic Transaction Reports
- Senate.gov: Financial Disclosures
"""

import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Rutas
RAW_DIR = "datos/raw"
os.makedirs(RAW_DIR, exist_ok=True)


def obtener_transacciones_house(año_inicio=2014, año_fin=None):
    """
    Descarga transacciones de la Cámara de Representantes.

    Fuente: https://disclosures.clerk.house.gov/PublicDisclosure/AnnualReports.aspx

    Returns:
        DataFrame con columnas: fecha, nombre, cargo, tipo, monto, titulo, partido, estado
    """
    if año_fin is None:
        año_fin = datetime.now().year

    logger.info(f"Descargando transacciones de House.gov ({año_inicio}-{año_fin})...")

    base_url = "https://disclosures.clerk.house.gov/PublicDisclosure/"

    todos_los_datos = []

    for año in range(año_inicio, año_fin + 1):
        logger.info(f"  Procesando año {año}...")

        # Intentar descargar el CSV del año
        # Los reportes anuales vienen en formato CSV
        csv_url = f"{base_url}AnnualReport_{año}.csv"

        try:
            response = requests.get(csv_url, timeout=30)
            if response.status_code == 200 and len(response.content) > 100:
                # Intentar parsing del CSV
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
                todos_los_datos.append(df)
                logger.info(f"    → {len(df)} registros de {año}")
            else:
                # Intentar con formato HTML
                logger.info(f"    → Intentando formato HTML...")
                _procesar_html_house(base_url, año, todos_los_datos)

        except Exception as e:
            logger.warning(f"    → Error con {año}: {e}")
            # Intentar formato HTML como backup
            try:
                _procesar_html_house(base_url, año, todos_los_datos)
            except Exception as e2:
                logger.warning(f"    → También falló HTML: {e2}")

        time.sleep(0.5)  # Ser respetuoso con el servidor

    if todos_los_datos:
        df_total = pd.concat(todos_los_datos, ignore_index=True)
        logger.info(f"Total: {len(df_total)} registros")
        return df_total
    else:
        logger.warning("No se pudieron obtener datos de House.gov")
        return pd.DataFrame()


def _procesar_html_house(base_url, año, todos_los_datos):
    """Procesa páginas HTML de House.gov como fallback."""
    # La página principal tiene links a los reportes
    pass  # Implementación dependería de la estructura exacta


def obtener_transacciones_senate(año_inicio=2014, año_fin=None):
    """
    Descarga transacciones del Senado.

    Fuente: https://www.senate.gov/legal/disclosure.htm

    Returns:
        DataFrame con columnas: fecha, nombre, cargo, tipo, monto, titulo, partido, estado
    """
    if año_fin is None:
        año_fin = datetime.now().year

    logger.info(f"Descargando transacciones de Senate.gov ({año_inicio}-{año_fin})...")

    # Senate usa un sistema diferente - disclosures直 接 en la web
    url = "https://www.senate.gov/legal/disclosure.htm"

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Buscar links a CSV o recursos
            links = soup.find_all('a', href=True)
            csv_links = [l['href'] for l in links if 'csv' in l['href'].lower() or 'disclosure' in l['href'].lower()]

            logger.info(f"  → Links encontrados: {len(csv_links)}")

            # Si hay CSVs disponibles, descargarlos
            if csv_links:
                todos_los_datos = []
                for csv_link in csv_links[:10]:  # Limitar a 10
                    try:
                        if not csv_link.startswith('http'):
                            csv_link = "https://www.senate.gov" + csv_link
                        df = pd.read_csv(csv_link)
                        todos_los_datos.append(df)
                    except:
                        pass

                if todos_los_datos:
                    return pd.concat(todos_los_datos, ignore_index=True)

    except Exception as e:
        logger.warning(f"Error accediendo a Senate.gov: {e}")

    return pd.DataFrame()


def descargar_datos_unusual_whales():
    """
    Descarga datos agregados de @unusual_whales (Twitter).

    Son más fáciles de procesar pero hay que scrape Twitter/API.

    Returns:
        DataFrame o None si no disponible
    """
    logger.info("Intentando unusual_whales...")

    # unusual_whales publica datos en GitHub también
    github_url = "https://raw.githubusercontent.com/unusual_whales/insider-trading-data/main/data/congress_trading.csv"

    try:
        df = pd.read_csv(github_url)
        logger.info(f"  → {len(df)} registros de unusual_whales")
        return df
    except Exception as e:
        logger.warning(f"  → No disponible: {e}")
        return pd.DataFrame()


def generar_dataset_completo():
    """
    Genera dataset combinando todas las fuentes.

    Returns:
        DataFrame unificado con columnas estandarizadas
    """
    logger.info("=" * 60)
    logger.info("GENERANDO DATASET COMPLETO")
    logger.info("=" * 60)

    # Intentar unusual_whales primero (más fácil)
    df_uw = descargar_datos_unusual_whales()

    if not df_uw.empty:
        logger.info(f"Datos de unusual_whales: {len(df_uw)} registros")
        df_uw.to_csv(f"{RAW_DIR}/congress_trading_raw.csv", index=False)

        # Estandarizar columnas
        df = _estandarizar_columnas(df_uw)
        df.to_csv(f"{RAW_DIR}/congress_trading_estandarizado.csv", index=False)
        return df

    # Fallback: House y Senate
    logger.info("Buscando en fuentes oficiales...")

    df_house = obtener_transacciones_house()
    df_senate = obtener_transacciones_senate()

    if df_house.empty and df_senate.empty:
        logger.error("No se pudieron obtener datos de ninguna fuente")
        return pd.DataFrame()

    # Combinar y estandarizar
    dfs = []
    if not df_house.empty:
        dfs.append(df_house)
    if not df_senate.empty:
        dfs.append(df_senate)

    if dfs:
        df_combined = pd.concat(dfs, ignore_index=True)
        df = _estandarizar_columnas(df_combined)
        df.to_csv(f"{RAW_DIR}/congress_trading_estandarizado.csv", index=False)
        return df

    return pd.DataFrame()


def _estandarizar_columnas(df):
    """
    Estandariza columnas de diferentes fuentes a formato común.

    Formato común:
    - fecha: datetime
    - nombre: str
    - tipo: 'compra' o 'venta'
    - monto: float
    - titulo: str (ticker o nombre del activo)
    - partido: 'D', 'R', 'I'
    - camara: 'senado' o 'camara'
    - estado: str (estado USA)
    - cargo: str (posición del congresista)
    """
    df = df.copy()

    # Mapear nombres de columnas comunes
    posibles_nombres = {
        'fecha': ['date', 'transaction_date', 'fecha', 'Date', 'TransactionDate'],
        'nombre': ['name', 'member', 'congress_member', 'Nombre', 'Member'],
        'tipo': ['type', 'transaction_type', 'tipo', 'Type', 'TransactionType', 'purchase', 'sale'],
        'monto': ['amount', 'monto', 'value', 'Amount', 'AmountReported'],
        'titulo': ['ticker', 'symbol', 'security', 'titulo', 'Ticker', 'Security'],
        'partido': ['party', 'partido', 'Party'],
        'camara': ['chamber', 'camara', 'Chamber'],
        'estado': ['state', 'Estado', 'State'],
        'cargo': ['position', 'title', 'cargo', 'Position', 'Title'],
    }

    # Renombrar columnas
    for col in df.columns:
        for standar, options in posibles_nombres.items():
            if col in options:
                df = df.rename(columns={col: standar})
                break

    # Estandarizar tipo
    if 'tipo' in df.columns:
        df['tipo'] = df['tipo'].astype(str).str.lower()
        df['tipo'] = df['tipo'].apply(lambda x: 'compra' if 'buy' in x or 'purchase' in x or 'compra' in x else ('venta' if 'sell' in x or 'sale' in x or 'venta' in x else 'unknown'))

    # Asegurar partido válido
    if 'partido' in df.columns:
        df['partido'] = df['partido'].str.upper()
        df['partido'] = df['partido'].apply(lambda x: x if x in ['D', 'R', 'I'] else 'I')

    # Asegurar cámara válida
    if 'camara' in df.columns:
        df['camara'] = df['camara'].str.lower()
        df['camara'] = df['camara'].apply(lambda x: 'senado' if 'senate' in str(x).lower() else 'camara')

    logger.info(f"Columnas estandarizadas: {list(df.columns)}")
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("DESCARGA DE DATOS - INSIDER TRADING CONGRESISTAS USA")
    print("=" * 60)
    print()

    df = generar_dataset_completo()

    if not df.empty:
        print(f"\n✓ Dataset generado: {len(df)} registros")
        print(f"  Columnas: {list(df.columns)}")
        print(f"\nPrimeras filas:")
        print(df.head())
    else:
        print("\n⚠️  No se pudieron obtener datos automáticamente.")
        print("    Opciones alternativas:")
        print("    1. Descargar manualmente desde House.gov/Senate.gov")
        print("    2. Seguir @unusual_whales para datos actualizados")
        print("    3. Buscar datasets en Kaggle")
