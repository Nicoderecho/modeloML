"""
Comisiones del Congreso - Datos de Committee Memberships y Leadership

Obtiene información sobre las comisiones y cargos de los congresistas
desde fuentes oficiales (Senate.gov, House.gov, Congress.gov API).
"""

import pandas as pd
import requests
from datetime import datetime
from typing import Dict, List, Optional
import logging
import json
import os
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = "datos/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Congress.gov API base URL
CONGRESS_API_BASE = "https://api.congress.gov/v3"
API_KEY = None  # Opcional, requiere registro en https://api.congress.gov/


def obtener_comisiones_senado(
    año_congreso: int = 117,
    usar_cache: bool = True
) -> Optional[Dict]:
    """
    Obtiene lista de comisiones del Senado para un Congreso específico.

    Fuente: Senate.gov API / XML

    Args:
        año_congreso: Número del Congreso (ej: 117 para 2021-2023)
        usar_cache: Si True, usa datos cacheados

    Returns:
        Dict con información de comisiones o None
    """
    cache_file = Path(CACHE_DIR) / f"senate_committees_{año_congreso}.json"

    if usar_cache and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Senate.gov no tiene REST API pública directa
    # Usamos datos estáticos de committee memberships
    logger.info(f"Obteniendo comisiones del Senado (Congreso {año_congreso})...")

    # Intentar Congress.gov API
    url = f"{CONGRESS_API_BASE}/committee/senate/{año_congreso}?format=json"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
            return data
    except Exception as e:
        logger.debug(f"Error con Congress API: {e}")

    # Fallback: datos hardcoded de comisiones principales
    comisiones_senado = {
        "committees": [
            {"name": "Agriculture, Nutrition, and Forestry", "code": "SSAF"},
            {"name": "Appropriations", "code": "SSAP"},
            {"name": "Armed Services", "code": "SSAS"},
            {"name": "Banking, Housing, and Urban Affairs", "code": "SSBK"},
            {"name": "Budget", "code": "SSBU"},
            {"name": "Commerce, Science, and Transportation", "code": "SSCM"},
            {"name": "Energy and Natural Resources", "code": "SSEG"},
            {"name": "Environment and Public Works", "code": "SSEP"},
            {"name": "Finance", "code": "SSFN"},
            {"name": "Foreign Relations", "code": "SSFR"},
            {"name": "Health, Education, Labor and Pensions", "code": "SSHR"},
            {"name": "Homeland Security and Governmental Affairs", "code": "SSGA"},
            {"name": "Judiciary", "code": "SSJU"},
            {"name": "Rules and Administration", "code": "SSRA"},
            {"name": "Small Business and Entrepreneurship", "code": "SBSE"},
            {"name": "Veterans' Affairs", "code": "SSVA"},
            {"name": "Intelligence (Select)", "code": "SSIN"},
            {"name": "Aging (Special)", "code": "SSF1"},
        ]
    }

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(comisiones_senado, f)

    return comisiones_senado


def obtener_comisiones_camara(
    año_congreso: int = 117,
    usar_cache: bool = True
) -> Optional[Dict]:
    """
    Obtiene lista de comisiones de la Cámara de Representantes.

    Args:
        año_congreso: Número del Congreso
        usar_cache: Si True, usa datos cacheados

    Returns:
        Dict con información de comisiones o None
    """
    cache_file = Path(CACHE_DIR) / f"house_committees_{año_congreso}.json"

    if usar_cache and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    logger.info(f"Obteniendo comisiones de la Cámara (Congreso {año_congreso})...")

    # Comisiones principales de la Cámara
    comisiones_camara = {
        "committees": [
            {"name": "Agriculture", "code": "HSAG"},
            {"name": "Appropriations", "code": "HSAP"},
            {"name": "Armed Services", "code": "HSAS"},
            {"name": "Budget", "code": "HSBU"},
            {"name": "Education and the Workforce", "code": "HSED"},
            {"name": "Energy and Commerce", "code": "HSIF"},
            {"name": "Ethics", "code": "HSET"},
            {"name": "Financial Services", "code": "HSBA"},
            {"name": "Foreign Affairs", "code": "HSFA"},
            {"name": "Homeland Security", "code": "HSHM"},
            {"name": "House Administration", "code": "HSHA"},
            {"name": "Judiciary", "code": "HSJU"},
            {"name": "Natural Resources", "code": "HSII"},
            {"name": "Oversight and Accountability", "code": "HSGO"},
            {"name": "Rules", "code": "HSRU"},
            {"name": "Science, Space, and Technology", "code": "HSSC"},
            {"name": "Small Business", "code": "HSSM"},
            {"name": "Transportation and Infrastructure", "code": "HSPW"},
            {"name": "Veterans' Affairs", "code": "HSVA"},
            {"name": "Ways and Means", "code": "HSWM"},
            {"name": "Intelligence (Permanent Select)", "code": "HSLC"},
        ]
    }

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(comisiones_camara, f)

    return comisiones_camara


def obtener_miembros_comision(
    camara: str,
    codigo_comision: str,
    año_congreso: int = 117
) -> Optional[List[Dict]]:
    """
    Obtiene lista de miembros de una comisión específica.

    Args:
        camara: "senate" o "house"
        codigo_comision: Código de la comisión (ej: "SSBK" para Senate Banking)
        año_congreso: Número del Congreso

    Returns:
        Lista de dicts con: name, party, state, role (chair/ranking/member)
    """
    logger.debug(f"  Obteniendo miembros de {camara} - {codigo_comision}")

    # Congress.gov API endpoint
    url = f"{CONGRESS_API_BASE}/committee_member/{camara}/{codigo_comision}/{año_congreso}?format=json"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return response.json().get("committeeMembers", [])
    except Exception as e:
        logger.debug(f"    Error API: {e}")

    # Fallback: retornar estructura vacía
    return None


def construir_mapa_comisiones_congresistas(
    df_transacciones: pd.DataFrame,
    año_inicio: int = 2014,
    año_fin: int = 2024
) -> Dict[str, Dict]:
    """
    Construye un mapa de comisiones y cargos para cada congresista
    en el DataFrame de transacciones.

    Args:
        df_transacciones: DataFrame con columna "Name"
        año_inicio: Año de inicio del análisis
        año_fin: Año de fin del análisis

    Returns:
        Dict keyed by congresista name con sus comisiones y cargos
    """
    logger.info("=" * 60)
    logger.info("CONSTRUYENDO MAPA DE COMISIONES DE CONGRESISTAS")
    logger.info("=" * 60)

    # Normalizar nombres
    nombres = df_transacciones["Name"].dropna().unique()
    nombres = [n.strip() for n in nombres if pd.notna(n)]

    logger.info(f"\nProcesando {len(nombres)} congresistas únicos...")

    congresistas_info = {}

    # Mapear años a números de Congreso
    # 113th Congress: 2013-2015
    # 114th Congress: 2015-2017
    # etc.
    def año_a_congreso(año: int) -> int:
        return 113 + (año - 2013) // 2

    congresos_a_procesar = list(range(
        año_a_congreso(año_inicio),
        año_a_congreso(año_fin) + 1
    ))

    logger.info(f"Congresos a procesar: {congresos_a_procesar}")

    # Comisiones de interés para análisis de insider trading
    comisiones_clave = {
        "senate": ["SSBK", "SSFN", "SSJU", "SSEG", "SSCM", "SSHR", "SSFR"],
        "house": ["HSBA", "HSWM", "HSJU", "HSIF", "HSFA", "HSED", "HSAS"]
    }

    # Mapear comisiones a sectores
    comision_a_sector = {
        "SSBK": "banking",      # Banking, Housing, Urban Affairs
        "SSFN": "finance",      # Finance
        "SSJU": "judiciary",    # Judiciary
        "SSEG": "energy",       # Energy and Natural Resources
        "SSCM": "commerce",     # Commerce, Science, Transportation
        "SSHR": "health",       # Health, Education, Labor, Pensions
        "SSFR": "foreign",      # Foreign Relations
        "HSBA": "banking",      # Financial Services
        "HSWM": "finance",      # Ways and Means
        "HSJU": "judiciary",    # Judiciary
        "HSIF": "energy",       # Energy and Commerce
        "HSFA": "foreign",      # Foreign Affairs
        "HSED": "education",    # Education and Workforce
        "HSAS": "defense",      # Armed Services
    }

    for idx, nombre in enumerate(nombres):
        if idx % 50 == 0:
            logger.info(f"  Procesando {idx}/{len(nombres)}: {nombre}")

        # Determinar cámara basándose en patrones de nombre
        # (En producción, esto debería venir de los datos originales)
        camara = determinar_camara(nombre, df_transacciones)

        congresistas_info[nombre] = {
            "name": nombre,
            "camara": camara,
            "partido": inferir_partido(nombre, df_transacciones),
            "comisiones": {},
            "liderazgo": [],
            "sectores": [],
        }

        # Para cada congreso, intentar obtener membresías
        for congreso in congresos_a_procesar:
            camara_key = "senate" if camara == "senado" else "house"
            comisiones_key = "senate" if camara == "senado" else "house"

            # Obtener membresías para comisiones clave
            for codigo in comisiones_clave.get(comisiones_key, []):
                miembros = obtener_miembros_comision(camara_key, codigo, congreso)

                if miembros:
                    # Buscar si este congresista está en la lista
                    for miembro in miembros:
                        miembro_nombre = limpiar_nombre(miembro.get("name", ""))
                        if nombres_equivalentes(nombre, miembro_nombre):
                            # Encontrado! Registrar comisión
                            sector = comision_a_sector.get(codigo, "other")

                            if codigo not in congresistas_info[nombre]["comisiones"]:
                                congresistas_info[nombre]["comisiones"][codigo] = {
                                    "nombre": miembro.get("committeeName", ""),
                                    "sector": sector,
                                    "congresos": []
                                }

                            rol = miembro.get("role", "member").lower()
                            if "chair" in rol:
                                rol = "chair"
                            elif "ranking" in rol:
                                rol = "ranking"
                            else:
                                rol = "member"

                            congresistas_info[nombre]["comisiones"][codigo]["congresos"].append({
                                "congreso": congreso,
                                "rol": rol
                            })

                            # Añadir sector si no existe
                            if sector not in congresistas_info[nombre]["sectores"]:
                                congresistas_info[nombre]["sectores"].append(sector)

    logger.info(f"\nMapa construido: {len(congresistas_info)} congresistas")

    # Guardar cache
    cache_file = Path(CACHE_DIR) / "congresistas_comisiones.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(congresistas_info, f, indent=2, ensure_ascii=False)

    logger.info(f"Cache guardado en: {cache_file}")

    return congresistas_info


def determinar_camara(
    nombre: str,
    df_transacciones: pd.DataFrame
) -> str:
    """
    Determina si un congresista es del Senado o Cámara.

    En producción, esto debería venir de los datos originales.
    Aquí usamos heurísticas basadas en patrones conocidos.
    """
    # Senadores conocidos (por contexto del proyecto)
    seno_conocidos = [
        "Sheldon Whitehouse", "A. Mitchell McConnell Jr.",
        "Nancy Pelosi", "Chuck Schumer", "John Boehner"
    ]

    # Por ahora, asumir todos son senadores (por SenatorCleaned.csv)
    return "senado"


def inferir_partido(
    nombre: str,
    df_transacciones: pd.DataFrame
) -> str:
    """
    Infiere el partido de un congresista.

    En producción, esto debería venir de una fuente externa.
    """
    # Partido conocido de algunos senadores
    partidos_conocidos = {
        "Sheldon Whitehouse": "D",
        "A. Mitchell McConnell Jr.": "R",
        "Nancy Pelosi": "D",
        "Chuck Schumer": "D",
        "John Boehner": "R",
    }

    for nombre_conocido, partido in partidos_conocidos.items():
        if nombre_conocido.lower() in nombre.lower():
            return partido

    return "U"  # Unknown


def limpiar_nombre(nombre: str) -> str:
    """Normaliza un nombre para comparación."""
    # Remover sufijos
    sufijos = ["Jr.", "Jr", "Sr.", "Sr", "III", "II", "IV"]
    for sufijo in sufijos:
        nombre = nombre.replace(sufijo, "")

    # Normalizar espacios y mayúsculas
    nombre = " ".join(nombre.split()).title()
    return nombre


def nombres_equivalentes(nombre1: str, nombre2: str) -> bool:
    """
    Verifica si dos nombres se refieren a la misma persona.

    Maneja variaciones como "Mitch McConnell" vs "A. Mitchell McConnell Jr."
    """
    n1 = limpiar_nombre(nombre1)
    n2 = limpiar_nombre(nombre2)

    # Comparar por palabras clave
    palabras1 = set(n1.lower().split())
    palabras2 = set(n2.lower().split())

    # Si comparten apellido y al menos una palabra más
    if len(palabras1.intersection(palabras2)) >= 2:
        return True

    # Caso especial: Mitch McConnell
    if "mcconnell" in palabras1 and "mcconnell" in palabras2:
        return True

    return False


def extraer_features_politicas(
    nombre: str,
    info_congresista: Dict
) -> Dict[str, int]:
    """
    Extrae features binarias/numéricas para el modelo ML.

    Args:
        nombre: Nombre del congresista
        info_congresista: Dict con comisiones y cargos

    Returns:
        Dict con features:
        - es_chair: 1 si preside alguna comisión
        - es_leader: 1 si es líder de partido
        - es_ranking: 1 si es ranking member
        - num_comisiones: número de comisiones
        - comision_banking: 1 si está en banking/finance
        - comision_energy: 1 si está en energy
        - comision_health: 1 si está en health
        - comision_judiciary: 1 si está en judiciary
        - comision_foreign: 1 si está en foreign relations
        - anos_en_congreso: años de servicio estimado
    """
    features = {
        "es_chair": 0,
        "es_leader": 0,
        "es_ranking": 0,
        "num_comisiones": 0,
        "comision_banking": 0,
        "comision_energy": 0,
        "comision_health": 0,
        "comision_judiciary": 0,
        "comision_foreign": 0,
        "anos_en_congreso": 0,
    }

    comisiones = info_congresista.get("comisiones", {})

    # Verificar roles
    for codigo, datos in comisiones.items():
        for periodo in datos.get("congresos", []):
            rol = periodo.get("rol", "member")

            if rol == "chair":
                features["es_chair"] = 1
            elif rol == "ranking":
                features["es_ranking"] = 1

        features["num_comisiones"] += 1

        # Verificar sectores
        sector = datos.get("sector", "")
        if sector in ["banking", "finance"]:
            features["comision_banking"] = 1
        elif sector == "energy":
            features["comision_energy"] = 1
        elif sector == "health":
            features["comision_health"] = 1
        elif sector == "judiciary":
            features["comision_judiciary"] = 1
        elif sector == "foreign":
            features["comision_foreign"] = 1

    # Años en congreso (estimado por número de congresos)
    num_congresos = len(set(
        p.get("congreso", 0)
        for datos in comisiones.values()
        for p in datos.get("congresos", [])
    ))
    features["anos_en_congreso"] = num_congresos * 2  # Cada congreso son 2 años

    return features


def generar_dataset_congresistas(
    df_transacciones: pd.DataFrame,
    ruta_output: str = "datos/processed/congresistas_info.csv"
) -> pd.DataFrame:
    """
    Genera dataset con información política de cada congresista.

    Args:
        df_transacciones: DataFrame con transacciones
        ruta_output: Ruta para guardar el dataset

    Returns:
        DataFrame con una fila por congresista con sus features políticas
    """
    logger.info("=" * 60)
    logger.info("GENERANDO DATASET DE INFORMACIÓN DE CONGRESISTAS")
    logger.info("=" * 60)

    # Construir mapa de comisiones
    mapa_comisiones = construir_mapa_comisiones_congresistas(df_transacciones)

    # Extraer features para cada congresista
    registros = []

    for nombre, info in mapa_comisiones.items():
        features = extraer_features_politicas(nombre, info)

        registro = {
            "name": nombre,
            "camara": info.get("camara", "senado"),
            "partido": info.get("partido", "U"),
            **features
        }

        registros.append(registro)

    df = pd.DataFrame(registros)

    # Guardar
    os.makedirs(os.path.dirname(ruta_output), exist_ok=True)
    df.to_csv(ruta_output, index=False)
    logger.info(f"Dataset guardado en: {ruta_output}")

    # Resumen
    logger.info(f"\nResumen:")
    logger.info(f"  Total congresistas: {len(df)}")
    logger.info(f"  Chairs: {df['es_chair'].sum()}")
    logger.info(f"  En Banking/Finance: {df['comision_banking'].sum()}")

    return df


if __name__ == "__main__":
    # Ejemplo de uso: cargar transacciones y generar info
    from enriquecer_precios import generar_dataset_enriquecido

    # Primero enriquecer con retornos
    df_transacciones = generar_dataset_enriquecido()

    # Luego generar información de congresistas
    df_info = generar_dataset_congresistas(df_transacciones)

    print("\n" + "=" * 60)
    print("INFORMACIÓN DE CONGRESISTAS")
    print("=" * 60)
    print(df_info.head(10))
