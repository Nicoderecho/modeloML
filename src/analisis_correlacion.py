"""
Análisis de Correlación - Tests Estadísticos para Trading de Congresistas

Realiza tests estadísticos (ANOVA, t-test, chi-square, regresión) para determinar
si existen correlaciones significativas entre características políticas y rendimiento.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
import logging
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def t_test_retornos_por_partido(
    df: pd.DataFrame,
    df_info: pd.DataFrame,
    min_n: int = 30
) -> Dict:
    """
    T-test para diferencia de retornos entre Demócratas y Republicanos.

    H0: No hay diferencia en retornos medios entre partidos
    H1: Hay diferencia significativa

    Args:
        df: DataFrame con transacciones enriquecidas
        df_info: DataFrame con información de partidos
        min_n: Mínimo de operaciones por grupo

    Returns:
        Dict con resultados del test:
        - t_statistic: estadístico t
        - p_value: valor p
        - significativo: bool si p < 0.05
        - efecto_cohen: d de Cohen (effect size)
        - conclusion: interpretación del resultado
    """
    logger.info("\n" + "=" * 60)
    logger.info("T-TEST: RETORNOS POR PARTIDO (D vs R)")
    logger.info("=" * 60)

    # Merge para obtener partido
    df_merged = df.merge(
        df_info[["name", "partido"]],
        left_on=df["Name"].str.strip().str.lower(),
        right_on=df_info["name"].str.strip().str.lower(),
        how="left"
    )
    df_merged["partido"] = df_merged["partido"].fillna("U")

    # Filtrar retornos válidos
    df_validos = df_merged[df_merged["retorno_porcentual"].notna()]

    # Separar por partido
    retornos_d = df_validos[df_validos["partido"] == "D"]["retorno_porcentual"]
    retornos_r = df_validos[df_validos["partido"] == "R"]["retorno_porcentual"]

    if len(retornos_d) < min_n or len(retornos_r) < min_n:
        return {
            "test": "t-test partido",
            "t_statistic": np.nan,
            "p_value": np.nan,
            "significativo": False,
            "efecto_cohen": np.nan,
            "conclusion": "Insuficientes datos para el test",
            "n_d": len(retornos_d),
            "n_r": len(retornos_r),
        }

    # T-test de dos muestras (asumiendo varianzas desiguales - Welch's t-test)
    t_stat, p_value = stats.ttest_ind(retornos_d, retornos_r, equal_var=False)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((retornos_d.std()**2 + retornos_r.std**()2) / 2)
    cohen_d = (retornos_d.mean() - retornos_r.mean()) / pooled_std

    # Interpretación
    if p_value < 0.01:
        conclusion = "Diferencia altamente significativa (p < 0.01)"
    elif p_value < 0.05:
        conclusion = "Diferencia significativa (p < 0.05)"
    else:
        conclusion = "No hay diferencia estadísticamente significativa"

    # Effect size interpretación
    if abs(cohen_d) < 0.2:
        conclusion += " - efecto trivial"
    elif abs(cohen_d) < 0.5:
        conclusion += " - efecto pequeño"
    elif abs(cohen_d) < 0.8:
        conclusion += " - efecto mediano"
    else:
        conclusion += " - efecto grande"

    logger.info(f"  Demócratas: n={len(retornos_d)}, mean={retornos_d.mean()*100:.2f}%")
    logger.info(f"  Republicanos: n={len(retornos_r)}, mean={retornos_r.mean()*100:.2f}%")
    logger.info(f"  t-statistic: {t_stat:.3f}")
    logger.info(f"  p-value: {p_value:.4f}")
    logger.info(f"  Cohen's d: {cohen_d:.3f}")
    logger.info(f"  Conclusión: {conclusion}")

    return {
        "test": "t-test partido",
        "t_statistic": t_stat,
        "p_value": p_value,
        "significativo": p_value < 0.05,
        "efecto_cohen": cohen_d,
        "conclusion": conclusion,
        "n_d": len(retornos_d),
        "n_r": len(retornos_r),
        "mean_d": retornos_d.mean(),
        "mean_r": retornos_r.mean(),
    }


def t_test_retornos_chair_vs_member(
    df: pd.DataFrame,
    df_info: pd.DataFrame,
    min_n: int = 30
) -> Dict:
    """
    T-test para diferencia de retornos entre Chairs y Members regulares.

    H0: Chairs no outperform a members
    H1: Chairs tienen retornos significativamente diferentes

    Args:
        df: DataFrame con transacciones enriquecidas
        df_info: DataFrame con información de cargos
        min_n: Mínimo de operaciones por grupo

    Returns:
        Dict con resultados del test
    """
    logger.info("\n" + "=" * 60)
    logger.info("T-TEST: RETORNOS CHAIR VS MEMBER")
    logger.info("=" * 60)

    # Obtener nombres de chairs y members
    chairs = df_info[df_info["es_chair"] == 1]["name"].str.lower().tolist()
    members = df_info[
        (df_info["es_chair"] == 0) &
        (df_info["es_ranking"] == 0)
    ]["name"].str.lower().tolist()

    # Filtrar retornos válidos
    df_validos = df[df["retorno_porcentual"].notna()].copy()
    df_validos["name_lower"] = df_validos["Name"].str.strip().str.lower()

    retornos_chairs = df_validos[df_validos["name_lower"].isin(chairs)]["retorno_porcentual"]
    retornos_members = df_validos[df_validos["name_lower"].isin(members)]["retorno_porcentual"]

    if len(retornos_chairs) < min_n or len(retornos_members) < min_n:
        return {
            "test": "t-test chair vs member",
            "t_statistic": np.nan,
            "p_value": np.nan,
            "significativo": False,
            "efecto_cohen": np.nan,
            "conclusion": "Insuficientes datos para el test",
        }

    # T-test
    t_stat, p_value = stats.ttest_ind(retornos_chairs, retornos_members, equal_var=False)

    # Effect size
    pooled_std = np.sqrt((retornos_chairs.std()**2 + retornos_members.std**()2) / 2)
    cohen_d = (retornos_chairs.mean() - retornos_members.mean()) / pooled_std

    # Interpretación
    if p_value < 0.01:
        conclusion = "Diferencia altamente significativa (p < 0.01)"
    elif p_value < 0.05:
        conclusion = "Diferencia significativa (p < 0.05)"
    else:
        conclusion = "No hay diferencia estadísticamente significativa"

    if abs(cohen_d) < 0.2:
        conclusion += " - efecto trivial"
    elif abs(cohen_d) < 0.5:
        conclusion += " - efecto pequeño"
    elif abs(cohen_d) < 0.8:
        conclusion += " - efecto mediano"
    else:
        conclusion += " - efecto grande"

    logger.info(f"  Chairs: n={len(retornos_chairs)}, mean={retornos_chairs.mean()*100:.2f}%")
    logger.info(f"  Members: n={len(retornos_members)}, mean={retornos_members.mean()*100:.2f}%")
    logger.info(f"  t-statistic: {t_stat:.3f}")
    logger.info(f"  p-value: {p_value:.4f}")
    logger.info(f"  Cohen's d: {cohen_d:.3f}")
    logger.info(f"  Conclusión: {conclusion}")

    return {
        "test": "t-test chair vs member",
        "t_statistic": t_stat,
        "p_value": p_value,
        "significativo": p_value < 0.05,
        "efecto_cohen": cohen_d,
        "conclusion": conclusion,
        "n_chairs": len(retornos_chairs),
        "n_members": len(retornos_members),
        "mean_chairs": retornos_chairs.mean(),
        "mean_members": retornos_members.mean(),
    }


def anova_retornos_por_sector(
    df: pd.DataFrame,
    df_info: pd.DataFrame,
    min_n: int = 20
) -> Dict:
    """
    ANOVA one-way para diferencia de retornos por sector de comisión.

    H0: Todos los sectores tienen el mismo retorno medio
    H1: Al menos un sector difiere

    Args:
        df: DataFrame con transacciones enriquecidas
        df_info: DataFrame con información de sectores
        min_n: Mínimo de operaciones por grupo

    Returns:
        Dict con resultados del ANOVA
    """
    logger.info("\n" + "=" * 60)
    logger.info("ANOVA: RETORNOS POR SECTOR DE COMISIÓN")
    logger.info("=" * 60)

    sectores = ["banking", "energy", "health", "judiciary", "foreign"]
    datos_sector = {}

    for sector in sectores:
        congresistas = df_info[df_info[f"comision_{sector}"] == 1]["name"].str.lower().tolist()
        retornos = df[
            (df["retorno_porcentual"].notna()) &
            (df["Name"].str.strip().str.lower().isin(congresistas))
        ]["retorno_porcentual"]

        if len(retornos) >= min_n:
            datos_sector[sector] = retornos

    if len(datos_sector) < 2:
        return {
            "test": "ANOVA sector",
            "f_statistic": np.nan,
            "p_value": np.nan,
            "significativo": False,
            "conclusion": "Insuficientes datos para el test",
        }

    # Preparar datos para ANOVA
    grupos = list(datos_sector.values())

    # One-way ANOVA
    f_stat, p_value = stats.f_oneway(*grupos)

    # Interpretación
    if p_value < 0.01:
        conclusion = "Diferencia altamente significativa entre sectores (p < 0.01)"
    elif p_value < 0.05:
        conclusion = "Diferencia significativa entre sectores (p < 0.05)"
    else:
        conclusion = "No hay diferencia significativa entre sectores"

    # Post-hoc test (Tukey) si es significativo
    post_hoc = None
    if p_value < 0.05:
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            # Preparar datos para Tukey
            all_data = []
            all_groups = []
            for sector, retornos in datos_sector.items():
                all_data.extend(retornos)
                all_groups.extend([sector] * len(retornos))

            tukey = pairwise_tukeyhsd(all_data, all_groups, alpha=0.05)
            post_hoc = str(tukey)
            conclusion += " - ver post-hoc Tukey para diferencias pairwise"
        except Exception as e:
            logger.debug(f"  Tukey test falló: {e}")

    logger.info(f"  Sectores analizados: {list(datos_sector.keys())}")
    for sector, retornos in datos_sector.items():
        logger.info(f"    {sector}: n={len(retornos)}, mean={retornos.mean()*100:.2f}%")
    logger.info(f"  F-statistic: {f_stat:.3f}")
    logger.info(f"  p-value: {p_value:.4f}")
    logger.info(f"  Conclusión: {conclusion}")

    return {
        "test": "ANOVA sector",
        "f_statistic": f_stat,
        "p_value": p_value,
        "significativo": p_value < 0.05,
        "conclusion": conclusion,
        "post_hoc": post_hoc,
        "sectores": {s: {"n": len(r), "mean": r.mean()} for s, r in datos_sector.items()},
    }


def chi_square_win_rate_por_partido(
    df: pd.DataFrame,
    df_info: pd.DataFrame,
    min_n: int = 30
) -> Dict:
    """
    Chi-square test para diferencia en win rate por partido.

    H0: Win rate es independiente del partido
    H1: Win rate depende del partido

    Args:
        df: DataFrame con transacciones enriquecidas
        df_info: DataFrame con información de partidos
        min_n: Mínimo de operaciones por grupo

    Returns:
        Dict con resultados del test
    """
    logger.info("\n" + "=" * 60)
    logger.info("CHI-SQUARE: WIN RATE POR PARTIDO")
    logger.info("=" * 60)

    # Merge para obtener partido
    df_merged = df.merge(
        df_info[["name", "partido"]],
        left_on=df["Name"].str.strip().str.lower(),
        right_on=df_info["name"].str.strip().str.lower(),
        how="left"
    )
    df_merged["partido"] = df_merged["partido"].fillna("U")

    # Filtrar retornos válidos
    df_validos = df_merged[df_merged["retorno_porcentual"].notna()]

    # Crear contingency table
    df_validos["win"] = (df_validos["retorno_porcentual"] > 0).astype(int)

    # Agrupar por partido
    observed = pd.crosstab(df_validos["partido"], df_validos["win"])

    if "D" not in observed.index or "R" not in observed.index:
        return {
            "test": "chi-square win rate",
            "chi2_statistic": np.nan,
            "p_value": np.nan,
            "significativo": False,
            "conclusion": "Insuficientes datos para el test",
        }

    contingency = observed.loc[["D", "R"]].values

    if contingency.min() < 5:
        return {
            "test": "chi-square win rate",
            "chi2_statistic": np.nan,
            "p_value": np.nan,
            "significativo": False,
            "conclusion": "Expected counts < 5, test no válido",
        }

    # Chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    # Interpretación
    if p_value < 0.01:
        conclusion = "Asociación altamente significativa (p < 0.01)"
    elif p_value < 0.05:
        conclusion = "Asociación significativa (p < 0.05)"
    else:
        conclusion = "No hay asociación significativa entre partido y win rate"

    # Win rates
    win_rate_d = df_validos[df_validos["partido"] == "D"]["win"].mean()
    win_rate_r = df_validos[df_validos["partido"] == "R"]["win"].mean()

    logger.info(f"  Demócratas win rate: {win_rate_d*100:.1f}%")
    logger.info(f"  Republicanos win rate: {win_rate_r*100:.1f}%")
    logger.info(f"  Chi2 statistic: {chi2:.3f}")
    logger.info(f"  p-value: {p_value:.4f}")
    logger.info(f"  Conclusión: {conclusion}")

    return {
        "test": "chi-square win rate",
        "chi2_statistic": chi2,
        "p_value": p_value,
        "significativo": p_value < 0.05,
        "conclusion": conclusion,
        "win_rate_d": win_rate_d,
        "win_rate_r": win_rate_r,
        "contingency_table": contingency.tolist(),
    }


def regresion_multiple_retornos(
    df: pd.DataFrame,
    df_info: pd.DataFrame
) -> Dict:
    """
    Regresión lineal múltiple para explicar retornos con variables políticas.

    Variables independientes:
    - es_chair: 1 si preside comisión
    - es_ranking: 1 si es ranking member
    - partido_R: 1 si Republicano
    - comision_banking: 1 si está en banking
    - comision_energy: 1 si está en energy
    - anos_en_congreso: años de servicio

    Args:
        df: DataFrame con transacciones enriquecidas
        df_info: DataFrame con información política

    Returns:
        Dict con resultados de la regresión
    """
    logger.info("\n" + "=" * 60)
    logger.info("REGRESIÓN MÚLTIPLE: EXPLICANDO RETORNOS")
    logger.info("=" * 60)

    # Preparar datos
    df_merged = df.merge(
        df_info,
        left_on=df["Name"].str.strip().str.lower(),
        right_on=df_info["name"].str.strip().str.lower(),
        how="left"
    )

    # Filtrar retornos válidos
    df_validos = df_merged[df_merged["retorno_porcentual"].notna()]

    if len(df_validos) < 100:
        return {
            "test": "regresión múltiple",
            "conclusion": "Insuficientes datos para regresión (min 100)",
            "rsquared": np.nan,
            "coeficientes": {},
        }

    # Preparar variables
    dependiente = "retorno_porcentual"
    independientes = [
        "es_chair",
        "es_ranking",
        "partido_R",  # Crear dummy variable
        "comision_banking",
        "comision_energy",
        "comision_health",
        "anos_en_congreso",
    ]

    # Crear dummy para partido
    df_validos["partido_R"] = (df_validos["partido"] == "R").astype(int)

    # Filtrar columnas existentes
    independientes = [c for c in independientes if c in df_validos.columns]

    # OLS regression
    try:
        formula = f"{dependiente} ~ {' + '.join(independientes)}"
        modelo = ols(formula, data=df_validos).fit()

        rsquared = model.rsquared
        rsquared_adj = model.rsquared_adj
        coeficientes = model.params
        p_values = model.pvalues
        aic = model.aic
        bic = model.bic

        # Interpretación
        conclusion = f"R² = {rsquared:.3f}, R² adj = {rsquared_adj:.3f}\n"
        conclusion += "Variables significativas (p < 0.05):\n"
        sig_vars = []
        for var, pval in p_values.items():
            if pval < 0.05 and var != "Intercept":
                conclusion += f"  - {var}: coef={coeficientes[var]:.4f}, p={pval:.4f}\n"
                sig_vars.append(var)

        if not sig_vars:
            conclusion += "  Ninguna variable es significativa"

        # Log output
        logger.info(f"  n = {len(df_validos)} observaciones")
        logger.info(f"  R² = {rsquared:.3f}")
        logger.info(f"  R² adj = {rsquared_adj:.3f}")
        logger.info(f"  AIC = {aic:.2f}")
        logger.info(f"  BIC = {bic:.2f}")
        logger.info(f"\n  Coeficientes:")
        for var in independientes:
            sig = "***" if p_values.get(var, 1) < 0.01 else ("**" if p_values.get(var, 1) < 0.05 else ("*" if p_values.get(var, 1) < 0.1 else ""))
            logger.info(f"    {var}: {coeficientes.get(var, np.nan):.6f} {sig}")
        logger.info(f"\n  Conclusión: {conclusion}")

        return {
            "test": "regresión múltiple",
            "n": len(df_validos),
            "rsquared": rsquared,
            "rsquared_adj": rsquared_adj,
            "aic": aic,
            "bic": bic,
            "coeficientes": coeficientes.to_dict(),
            "p_values": p_values.to_dict(),
            "conclusion": conclusion,
            "variables_significativas": sig_vars,
        }

    except Exception as e:
        logger.error(f"  Error en regresión: {e}")
        return {
            "test": "regresión múltiple",
            "conclusion": f"Error en regresión: {e}",
            "rsquared": np.nan,
            "coeficientes": {},
        }


def generar_reporte_correlacion_completo(
    df_transacciones: pd.DataFrame,
    df_info: pd.DataFrame,
    ruta_output: str = "datos/processed/reporte_correlacion.json"
) -> Dict:
    """
    Genera reporte completo de todos los tests estadísticos.

    Args:
        df_transacciones: DataFrame con transacciones enriquecidas
        df_info: DataFrame con información política
        ruta_output: Ruta para guardar reporte JSON

    Returns:
        Dict con resultados de todos los tests
    """
    logger.info("=" * 60)
    logger.info("GENERANDO REPORTE DE CORRELACIÓN ESTADÍSTICA")
    logger.info("=" * 60)

    resultados = {}

    # 1. T-test por partido
    logger.info("\n[1/5] T-test partido...")
    resultados["t_test_partido"] = t_test_retornos_por_partido(df_transacciones, df_info)

    # 2. T-test chair vs member
    logger.info("[2/5] T-test chair vs member...")
    resultados["t_test_chair"] = t_test_retornos_chair_vs_member(df_transacciones, df_info)

    # 3. ANOVA por sector
    logger.info("[3/5] ANOVA por sector...")
    resultados["anova_sector"] = anova_retornos_por_sector(df_transacciones, df_info)

    # 4. Chi-square win rate
    logger.info("[4/5] Chi-square win rate...")
    resultados["chi_square"] = chi_square_win_rate_por_partido(df_transacciones, df_info)

    # 5. Regresión múltiple
    logger.info("[5/5] Regresión múltiple...")
    resultados["regresion"] = regresion_multiple_retornos(df_transacciones, df_info)

    # Guardar reporte
    import json
    os.makedirs(os.path.dirname(ruta_output), exist_ok=True)

    # Convertir numpy types a Python types para JSON
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
        return obj

    resultados_serializados = serialize(resultados)

    with open(ruta_output, "w", encoding="utf-8") as f:
        json.dump(resultados_serializados, f, indent=2)

    logger.info(f"\nReporte guardado en: {ruta_output}")

    # Resumen final
    logger.info("\n" + "=" * 60)
    logger.info("RESUMEN DE HALLAZGOS ESTADÍSTICOS")
    logger.info("=" * 60)

    hallazgos_sig = []
    for test_name, resultado in resultados.items():
        if resultado.get("significativo", False):
            hallazgos_sig.append(test_name)

    if hallazgos_sig:
        logger.info(f"\nTests significativos (p < 0.05): {hallazgos_sig}")
    else:
        logger.info("\nNo se encontraron diferencias estadísticamente significativas")

    return resultados


if __name__ == "__main__":
    # Ejecutar análisis de correlación
    from enriquecer_precios import generar_dataset_enriquecido
    from comisiones_congreso import generar_dataset_congresistas

    # Cargar datos
    df_transacciones = generar_dataset_enriquecido()
    df_info = generar_dataset_congresistas(df_transacciones)

    # Generar reporte
    resultados = generar_reporte_correlacion_completo(df_transacciones, df_info)

    print("\n" + "=" * 60)
    print("RESULTADOS DE TESTS ESTADÍSTICOS")
    print("=" * 60)

    for test_name, resultado in resultados.items():
        print(f"\n{test_name}:")
        print(f"  Conclusión: {resultado.get('conclusion', 'N/A')}")
