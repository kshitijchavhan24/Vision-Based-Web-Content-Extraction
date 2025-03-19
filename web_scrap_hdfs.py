import os
import time
import re
import math
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import WebDriverException
from hdfs import InsecureClient

def initialize_webdriver(chrome_driver_path):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    # Set the viewport to 1920 x 1080
    chrome_options.add_argument("--window-size=1920,1080")
    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.set_page_load_timeout(300)
    return driver

def fetch_with_retries(driver, url, retries=3, delay=5):
    for attempt in range(retries):
        try:
            driver.get(url)
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {url}. Retrying in {delay} seconds...")
            time.sleep(delay)
    print(f"Failed to fetch {url} after {retries} retries.")
    return False

def capture_viewport_screenshots(driver):
    """
    Scrolls through the page by viewport-height increments (1920x1080)
    and captures a screenshot for each segment. Returns a list of PNG bytes.
    """
    # Start at the top
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(1)
    
    viewport_height = driver.execute_script("return window.innerHeight")
    document_height = driver.execute_script("return document.body.scrollHeight")
    num_viewports = math.ceil(document_height / viewport_height)
    
    screenshots = []
    for i in range(num_viewports):
        position = i * viewport_height
        driver.execute_script("window.scrollTo(0, arguments[0]);", position)
        time.sleep(1)  # Allow time for scrolling/lazy-loading
        screenshot = driver.get_screenshot_as_png()  # Returns binary PNG data
        screenshots.append(screenshot)
    return screenshots

def find_next_page(driver):
    """
    Searches for a clickable <a> or <button> element that likely points to the next page.
    It looks for common keywords.
    """
    possible_terms = ["next", "load more", "page"]
    elements = driver.find_elements(By.XPATH, "//a | //button")
    for elem in elements:
        try:
            text = elem.text.strip().lower()
            if any(term in text for term in possible_terms):
                return elem
        except Exception:
            continue
    return None

def sanitize_folder_name(s):
    """
    Replaces any characters not safe for folder names with underscores.
    """
    return re.sub(r"[^\w\-_.]", "_", s)

def main():
    chrome_driver_path = "/usr/local/bin/chromedriver"  # Adjust this path if needed

    # Initialize HDFS client (ensure your HDFS NameNode is running at the given URL)
    hdfs_client = InsecureClient('http://localhost:9870', user='kc', timeout=300)
    try:
        hdfs_client.makedirs('/scraped_images')
    except Exception as e:
        print(f"HDFS directory creation note: {e}")

    # Ask for URLs (comma-separated input)
    urls = input("Enter the URLs to scrape (comma-separated): ").strip().split(",")
    urls = [url.strip() for url in urls if url.strip()]

    driver = initialize_webdriver(chrome_driver_path)
    try:
        for url in urls:
            # Create a folder name based on the URL's domain and path
            parsed = urlparse(url)
            domain = parsed.netloc.replace(".", "_")
            path = parsed.path.strip("/")
            if not path:
                path = "home"
            else:
                path = path.replace("/", "_")
            url_folder = sanitize_folder_name(f"{domain}_{path}")
            try:
                hdfs_client.makedirs(f"/scraped_images/{url_folder}")
            except Exception as e:
                print(f"HDFS subdirectory creation note: {e}")

            page_id = 1
            next_url = url
            while next_url:
                print(f"\nScraping page {page_id} of {url}...")
                if not fetch_with_retries(driver, next_url):
                    break

                time.sleep(1)  # Brief pause after page load

                # Capture viewport screenshots (each as 1920x1080)
                screenshots = capture_viewport_screenshots(driver)

                # Create an HDFS directory for this page
                hdfs_page_dir = f"/scraped_images/{url_folder}/page_{page_id}"
                try:
                    hdfs_client.makedirs(hdfs_page_dir)
                except Exception as e:
                    print(f"HDFS page directory creation note: {e}")

                # Upload each screenshot directly to HDFS
                for idx, screenshot_data in enumerate(screenshots):
                    file_name = f"screenshot_{idx+1}.png"
                    hdfs_path = f"{hdfs_page_dir}/{file_name}"
                    try:
                        with hdfs_client.write(hdfs_path, overwrite=True) as writer:
                            writer.write(screenshot_data)
                        print(f"Uploaded screenshot to {hdfs_path}")
                    except Exception as upload_err:
                        print(f"Error uploading to HDFS: {upload_err}")

                # Look for a pagination element to navigate to the next page
                next_button = find_next_page(driver)
                if next_button:
                    try:
                        ActionChains(driver).move_to_element(next_button).click(next_button).perform()
                        time.sleep(2)
                        next_url = driver.current_url
                        page_id += 1
                    except WebDriverException:
                        print("Unable to navigate to the next page.")
                        break
                else:
                    print("No pagination button found. Assuming last page.")
                    break
            print(f"Finished scraping {url}.\n")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
