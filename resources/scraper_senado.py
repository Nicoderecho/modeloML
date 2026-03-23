"""
Scraper de Transacciones del Congreso (Senado)
Combina mejoras del notebook con la robustez del script original
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import os
from datetime import datetime

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
WEBSITE = "https://efdsearch.senate.gov/"

# Fecha de inicio para filtrar (formato MM/DD/YYYY)
# Dejar en None para no filtrar por fecha
START_DATE = "03/01/2026"  # Ejemplo: "07/01/2024"

# Número de páginas a recorrer (None = todas las páginas disponibles)
MAX_PAGES = None

# Número de entradas por página (25, 50, 75 o 100)
ENTRIES_PER_PAGE = 100

# Ruta de salida (relativa al script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'CongressInvestments_New.csv')

# =============================================================================


def setup_browser():
    """Configura y devuelve el navegador Chrome con opciones optimizadas."""
    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-logging"])
    options.add_argument("--log-level=3")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--headless")  # Modo sin ventana

    from selenium.webdriver.chrome.service import Service
    service = Service(ChromeDriverManager().install())
    browser = webdriver.Chrome(service=service, options=options)
    browser.maximize_window()
    return browser


def navigate_to_search(browser):
    """Navega al sitio y configura los criterios de búsqueda."""
    browser.get(WEBSITE)

    # Aceptar términos
    WebDriverWait(browser, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//*[@id='agree_statement']"))
    ).click()
    time.sleep(1)

    # Seleccionar "Senate Periodic Transactions"
    browser.find_element(By.XPATH,
        "//*[contains(concat(' ', @class, ' '), concat(' ', 'form-check-input', ' '))]").click()
    browser.find_element(By.XPATH, "//*[@id='reportTypeLabelPtr']").click()
    time.sleep(1)

    # Añadir filtro de fecha si está configurado
    if START_DATE:
        date_input = browser.find_element(By.XPATH,
            '/html/body/div[1]/main/div/div/div[5]/div/form/fieldset[4]/div/div/div/div[1]/span/input')
        date_input.send_keys(START_DATE)
        print(f"Filtrando desde: {START_DATE}")

    # Hacer clic en buscar
    browser.find_element(By.XPATH, "//*[@id='searchForm']/div/button").click()
    time.sleep(2)


def set_entries_per_page(browser):
    """Configura el número de entradas por página."""
    try:
        # Seleccionar opción de 100 entradas por página
        dropdown = browser.find_element(By.XPATH,
            "/html/body/div[1]/main/div/div/div[6]/div/div/div/div[3]/div[1]/div/label/select")

        # Mapeo de valores al índice de opción
        entries_map = {25: 1, 50: 2, 75: 3, 100: 4}
        option_index = entries_map.get(ENTRIES_PER_PAGE, 4)

        dropdown.find_element(By.XPATH, f"./option[{option_index}]").click()
        time.sleep(2)
        print(f"Configurado a {ENTRIES_PER_PAGE} entradas por página")
    except Exception as e:
        print(f"No se pudo cambiar entradas por página: {e}")


def extract_transactions_from_report(browser, df):
    """Extrae transacciones de un reporte individual."""
    try:
        # Obtener número de transacciones en este reporte
        num_transactions = int(browser.find_element(By.XPATH,
            '//*[@id="content"]/div/div/section/div/div/table/tbody/tr[1]/td[1]').text)

        # Extraer nombre del senador
        name = browser.find_element(By.XPATH, '//*[@id="content"]/div/div/div[2]/div[1]/h2').text

        for individual in range(1, num_transactions + 1):
            try:
                row_data = {
                    'Name': name,
                    'Transaction Date': browser.find_element(By.XPATH,
                        f'//*[@id="content"]/div/div/section/div/div/table/tbody/tr[{individual}]/td[2]').text,
                    'Owner': browser.find_element(By.XPATH,
                        f'//*[@id="content"]/div/div/section/div/div/table/tbody/tr[{individual}]/td[3]').text,
                    'Ticker': browser.find_element(By.XPATH,
                        f'//*[@id="content"]/div/div/section/div/div/table/tbody/tr[{individual}]/td[4]').text,
                    'Asset Name': browser.find_element(By.XPATH,
                        f'//*[@id="content"]/div/div/section/div/div/table/tbody/tr[{individual}]/td[5]').text,
                    'Asset Type': browser.find_element(By.XPATH,
                        f'//*[@id="content"]/div/div/section/div/div/table/tbody/tr[{individual}]/td[6]').text,
                    'Type': browser.find_element(By.XPATH,
                        f'//*[@id="content"]/div/div/section/div/div/table/tbody/tr[{individual}]/td[7]').text,
                    'Amount': browser.find_element(By.XPATH,
                        f'//*[@id="content"]/div/div/section/div/div/table/tbody/tr[{individual}]/td[8]').text,
                    'Comment': browser.find_element(By.XPATH,
                        f'//*[@id="content"]/div/div/section/div/div/table/tbody/tr[{individual}]/td[9]').text
                }

                df.loc[len(df)] = row_data

            except Exception as e:
                print(f"Error extrayendo transacción {individual}: {e}")
                continue

    except Exception as e:
        print(f"Error procesando reporte: {e}")


def process_page(browser, df, entries_per_page):
    """Procesa una página de resultados."""
    processed = 0
    skipped = 0

    # Detectar dinámicamente cuántas filas hay en la tabla
    rows = browser.find_elements(By.XPATH, '//*[@id="filedReports"]/tbody/tr')
    num_rows = len(rows)

    if num_rows == 0:
        print("No se encontraron filas en la tabla")
        return 0, 0

    print(f"Filas detectadas en la página: {num_rows}")

    for count in range(1, min(num_rows + 1, entries_per_page + 1)):
        try:
            # Verificar si la fila tiene datos (no es encabezado)
            caps = browser.find_element(By.XPATH,
                f'//*[@id="filedReports"]/tbody/tr[{count}]/td[1]').text

            if caps.isupper():
                skipped += 1
                continue

            # Abrir reporte
            button = browser.find_element(By.XPATH,
                f'/html/body/div[1]/main/div/div/div[6]/div/div/div/table/tbody/tr[{count}]/td[4]/a')
            browser.execute_script("arguments[0].click();", button)
            time.sleep(2)

            # Cambiar a nueva pestaña
            browser.switch_to.window(browser.window_handles[1])
            time.sleep(1)

            # Extraer datos
            extract_transactions_from_report(browser, df)
            processed += 1

            # Cerrar pestaña y volver
            browser.close()
            browser.switch_to.window(browser.window_handles[0])
            time.sleep(0.5)

        except Exception as e:
            print(f"Error en entrada {count}: {e}")
            # Asegurar que volvemos a la ventana principal
            if len(browser.window_handles) > 1:
                browser.close()
                browser.switch_to.window(browser.window_handles[0])
            continue

    return processed, skipped


def get_next_page(browser):
    """Intenta ir a la siguiente página. Retorna True si tiene éxito."""
    try:
        next_button = browser.find_element(By.XPATH,
            '/html/body/div[1]/main/div/div/div[6]/div/div/div/div[3]/div[2]/div/a[2]')

        # Verificar si el botón está deshabilitado (última página)
        aria_disabled = next_button.get_attribute('aria-disabled')
        if aria_disabled == 'true':
            return False

        next_button.click()
        time.sleep(3)
        return True
    except Exception as e:
        print(f"No se pudo navegar a siguiente página: {e}")
        return False


def get_current_page_number(browser):
    """Obtiene el número de página actual."""
    try:
        page_text = browser.find_element(By.XPATH,
            '//*[contains(concat(" ", @class, " "), concat(" ", "current", " "))]').text
        return int(page_text)
    except:
        return 1


def scrape():
    """Función principal de scraping."""
    start_time = time.time()

    # Inicializar DataFrame
    columns = ['Name', 'Transaction Date', 'Owner', 'Ticker', 'Asset Name',
               'Asset Type', 'Type', 'Amount', 'Comment']
    df = pd.DataFrame(columns=columns)

    print("=" * 60)
    print("SCRAPER DE TRANSACCIONES DEL SENADO")
    print("=" * 60)
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Fecha mínima: {START_DATE if START_DATE else 'Todas'}")
    print(f"Entradas por página: {ENTRIES_PER_PAGE}")
    print(f"Máximo páginas: {MAX_PAGES if MAX_PAGES else 'Ilimitado'}")
    print("=" * 60)

    browser = None
    try:
        browser = setup_browser()
        navigate_to_search(browser)
        set_entries_per_page(browser)

        page_number = get_current_page_number(browser)
        total_pages_processed = 0

        while True:
            print(f"\n--- Procesando página {page_number} ---")
            processed, skipped = process_page(browser, df, ENTRIES_PER_PAGE)

            elapsed = time.time() - start_time
            print(f"Entradas procesadas: {processed}, Saltadas: {skipped}")
            print(f"Total transacciones: {len(df)}")
            print(f"Tiempo transcurrido: {elapsed:.1f}s")

            total_pages_processed += 1

            # Verificar si alcanzamos el máximo de páginas
            if MAX_PAGES and total_pages_processed >= MAX_PAGES:
                print(f"\nMáximo de páginas ({MAX_PAGES}) alcanzado.")
                break

            # Intentar ir a siguiente página
            if not get_next_page(browser):
                print("\nNo hay más páginas.")
                break

            page_number += 1

        # Guardar resultados
        print(f"\n{'=' * 60}")
        print("RESUMEN FINAL")
        print(f"{'=' * 60}")
        print(f"Total de transacciones extraídas: {len(df)}")
        print(f"Páginas procesadas: {total_pages_processed}")
        print(f"Tiempo total: {(time.time() - start_time) / 60:.2f} minutos")

        # Guardar CSV
        output_dir = os.path.dirname(OUTPUT_PATH)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        df.to_csv(OUTPUT_PATH, index=False)
        print(f"\nArchivo guardado en: {OUTPUT_PATH}")
        print(f"Tamaño: {os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB")

        return df

    except Exception as e:
        print(f"\nERROR CRÍTICO: {e}")
        # Guardar progreso parcial
        if len(df) > 0:
            partial_path = OUTPUT_PATH.replace('.csv', '_partial.csv')
            df.to_csv(partial_path, index=False)
            print(f"Progreso parcial guardado en: {partial_path}")
        raise

    finally:
        if browser:
            browser.quit()
            print("\nNavegador cerrado.")


if __name__ == "__main__":
    scrape()
