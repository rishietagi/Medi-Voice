from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
import traceback

CHROME_DRIVER_PATH = r"C:\Users\Rishi S Etagi\Downloads\chromedriver-win64\chromedriver-win64\chromedriver.exe"

AUDIO_FILE = os.path.abspath("sample_audio.wav")
assert os.path.exists(AUDIO_FILE), f"Audio file not found: {AUDIO_FILE}"

options = Options()
options.add_argument("--headless") 
options.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(service=Service(CHROME_DRIVER_PATH), options=options)

try:
    driver.get("http://localhost:8501")

    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
    )

    upload_input = driver.find_element(By.CSS_SELECTOR, "input[type='file']")
    upload_input.send_keys(AUDIO_FILE)

    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Heart Rate')]"))
    )

    print("Test Passed")

except Exception as e:
    print("Test Failed")
    traceback.print_exc()
    driver.save_screenshot("debug_screenshot.png")
    print("Screenshot saved ")

finally:
    driver.quit()
