import time
import random
import csv
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Set up the Selenium driver
driver = webdriver.Chrome()

from utilsLoad import ensure_directory_exists, save_metadata
from classPuzzle import MetaData

# Number of puzzles to download
num_puzzles = 5

# Define the minimum and maximum values for each input field
ranges = {
    "_seed": (1, 1000),
    "_tabsize": (15, 20),
    "_jitter": (0, 7),
    "xn": (5, 6),
    "yn": (5, 6),
    "width": (100, 300),
    "height": (100, 300),
}


for i in range(num_puzzles):
    try:
        driver.get("https://puzzle.telegnom.org/")

        # Wait for the page to load completely
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "button"))
        )

        # Dictionary to hold the values used
        used_values = {}

        # Set random values for each input field
        for input_id, (min_val, max_val) in ranges.items():
            element = driver.find_element(By.ID, input_id)
            element.clear()
            value = random.randint(min_val, max_val)
            element.send_keys(str(value))
            used_values[input_id] = value

        # Convert used values to MetaData object and save it
        metadata = MetaData(**used_values)
        metadata_path = f"Puzzles/Shuffled/jigsaw{i}/"
        ensure_directory_exists(metadata_path)
        metadata_file_path = os.path.join(metadata_path, "puzzle_meta_data.json")
        save_metadata(metadata, metadata_file_path)

        # Find the button for generating the puzzle and click it
        generate_button = driver.find_element(
            By.XPATH, '//button[text()="Download SVG"]'
        )
        generate_button.click()

        puzzle_name = f"jigsaw{i}.svg"
        print(f"Imported {puzzle_name}")
        # Optional delay between requests
        time.sleep(2)

    except Exception as e:
        print(f"An error occurred: {e}")
        break

# Close the browser
driver.quit()
