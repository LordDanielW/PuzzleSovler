import time
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Set up the Selenium driver
driver = webdriver.Chrome()

# Number of puzzles to download
num_puzzles = 10

# Define the minimum and maximum values for each input field


ranges = {
    "_seed": (1, 1000),
    "_tabsize": (15, 20),
    "_jitter": (0, 7),
    "xn": (5, 10),
    "yn": (5, 10),
    "width": (100, 500),  # Assuming the width is in pixels or similar units
    "height": (100, 500),  # Assuming the height is in pixels or similar units
}

for i in range(num_puzzles):
    try:
        driver.get("https://puzzle.telegnom.org/")

        # Wait for the page to load completely
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "button"))
        )  # Waiting for the button to be present

        # Set random values for each input field
        for input_id, (min_val, max_val) in ranges.items():
            element = driver.find_element(By.ID, input_id)
            element.clear()
            # Generate a random integer within the specified range and input it into the field
            element.send_keys(str(random.randint(min_val, max_val)))

        # Find the button for generating the puzzle and click it
        generate_button = driver.find_element(
            By.XPATH, '//button[text()="Download SVG"]'
        )
        generate_button.click()

        # Optional delay between requests to avoid overloading the server
        time.sleep(4)

    except Exception as e:
        print(f"An error occurred: {e}")
        break

# Close the browser
driver.quit()
