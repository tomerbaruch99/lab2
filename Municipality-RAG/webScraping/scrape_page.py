"""
Scrape individual pages from municipality websites.
Supports various page types: articles, services, announcements, etc.
"""

import ast
import time 
import json
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException, InvalidSessionIdException, StaleElementReferenceException, TimeoutException

def scrape_article_page(wd, link: str, path_parts: list, timeout: int):
    """
    Scrape an article page from municipality website.
    
    Args:
        wd: WebDriver instance
        link: URL to scrape
        path_parts: Path parts from URL
        timeout: Timeout for page load
    
    Returns:
        Dictionary with scraped data
    """
    assert link != "", "no link was provided"
    wd.get(link)
    time.sleep(timeout)
    
    try:
        main_element = wd.find_element(By.TAG_NAME, "main")
    except NoSuchElementException:
        main_element = wd.find_element(By.TAG_NAME, "body")

    # Extract title
    try:
        title_element = main_element.find_element(By.CSS_SELECTOR, ".title, h1, [class*='title']")
        title = title_element.text
    except NoSuchElementException:
        title = None

    # Extract subtitle
    try:
        subtitle_element = main_element.find_element(By.CSS_SELECTOR, ".subtitle, .general_subtitle, h2")
        subtitle = subtitle_element.text
    except NoSuchElementException:
        subtitle = None

    # Extract all text content
    article_text = ""
    try:
        text_elements = main_element.find_elements(By.CSS_SELECTOR, "p, .content, [class*='text'], [class*='content']")
        for text in text_elements:
            text_content = text.text.strip()
            if text_content and text_content not in article_text:
                article_text += text_content + "\n"
    except NoSuchElementException:
        article_text = ""

    # Extract links
    links = []
    try:
        link_elements = main_element.find_elements(By.CSS_SELECTOR, "a[href]")
        for elem in link_elements:
            href = elem.get_attribute("href")
            if href and href.startswith("http"):
                links.append(href)
    except NoSuchElementException:
        links = []

    # Extract image links
    image_links = []
    try:
        image_elements = main_element.find_elements(By.TAG_NAME, "img")
        for elem in image_elements:
            src = elem.get_attribute("src")
            if src:
                if src.startswith("http"):
                    image_links.append(src)
                elif src.startswith("/"):
                    # Relative URL - construct full URL
                    from urllib.parse import urljoin
                    image_links.append(urljoin(link, src))
    except NoSuchElementException:
        image_links = []

    return {
        "page_link": link,
        "categories": [item for item in path_parts if not item.isdigit()],
        "title": title,
        "subtitle": subtitle,
        "article_text": article_text,
        "links": links,
        "image_links": image_links
    }

def scrape_service_page(wd, link: str, path_parts: list, timeout: int):
    """
    Scrape a service page from municipality website.
    
    Args:
        wd: WebDriver instance
        link: URL to scrape
        path_parts: Path parts from URL
        timeout: Timeout for page load
    
    Returns:
        Dictionary with scraped data
    """
    assert link != "", "no link was provided"
    wd.get(link)
    time.sleep(timeout)
    
    try:
        main_element = wd.find_element(By.TAG_NAME, "main")
    except NoSuchElementException:
        main_element = wd.find_element(By.TAG_NAME, "body")

    # Extract title
    try:
        title_element = main_element.find_element(By.CSS_SELECTOR, ".title, h1, [class*='title']")
        title = title_element.text
    except NoSuchElementException:
        title = None

    # Extract description
    try:
        description_element = main_element.find_element(By.CSS_SELECTOR, ".description, .text, [class*='description']")
        description = description_element.text
    except NoSuchElementException:
        description = ""

    # Extract service details
    details = []
    try:
        detail_elements = main_element.find_elements(By.CSS_SELECTOR, ".detail, .info, [class*='detail']")
        for elem in detail_elements:
            details.append(elem.text.strip())
    except NoSuchElementException:
        pass

    return {
        "page_link": link,
        "categories": [item for item in path_parts if not item.isdigit()],
        "title": title,
        "description": description,
        "details": details
    }

def scrape_announcement_page(wd, link: str, path_parts: list, timeout: int):
    """
    Scrape an announcement page from municipality website.
    
    Args:
        wd: WebDriver instance
        link: URL to scrape
        path_parts: Path parts from URL
        timeout: Timeout for page load
    
    Returns:
        Dictionary with scraped data
    """
    assert link != "", "no link was provided"
    wd.get(link)
    time.sleep(timeout)
    
    try:
        main_element = wd.find_element(By.TAG_NAME, "main")
    except NoSuchElementException:
        main_element = wd.find_element(By.TAG_NAME, "body")

    # Extract title
    try:
        title_element = main_element.find_element(By.CSS_SELECTOR, ".title, h1, [class*='title']")
        title = title_element.text
    except NoSuchElementException:
        title = None

    # Extract date
    try:
        date_element = main_element.find_element(By.CSS_SELECTOR, ".date, [class*='date'], time")
        date = date_element.text
    except NoSuchElementException:
        date = None

    # Extract content
    content = ""
    try:
        content_elements = main_element.find_elements(By.CSS_SELECTOR, "p, .content, [class*='content']")
        for elem in content_elements:
            text = elem.text.strip()
            if text:
                content += text + "\n"
    except NoSuchElementException:
        content = ""

    return {
        "page_link": link,
        "categories": [item for item in path_parts if not item.isdigit()],
        "title": title,
        "date": date,
        "content": content
    }

def scraper_manager(page_type: str):
    """
    Return the appropriate scraper function based on page type.
    
    Args:
        page_type: Type of page ("article", "service", "announcement")
    
    Returns:
        Scraper function
    """
    assert page_type != "", "Must give page_type"
    if page_type == "article":
        return scrape_article_page
    elif page_type == "service":
        return scrape_service_page
    elif page_type == "announcement":
        return scrape_announcement_page
    else:
        # Default to article scraper
        return scrape_article_page

if __name__ == "__main__":
    # Setup WebDriver
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    # For local development, use ChromeDriverManager or specify path
    # For Docker, use the paths specified in Dockerfile
    try:
        wd = webdriver.Chrome(options=chrome_options)
    except Exception as e:
        print(f"Error initializing WebDriver: {e}")
        print("Make sure ChromeDriver is installed and in PATH")
        exit(1)
    
    page_type = "article"  # or "service", "announcement"
    df = pd.read_csv(f"data/haifa_{page_type}_links.csv")
    
    scraper = scraper_manager(page_type)
    page_data = []
    
    for index, row in tqdm(df.iterrows(), total=len(df)):
        print(row["link"])
        try:
            path_parts = ast.literal_eval(row["path_parts"]) if isinstance(row["path_parts"], str) else row["path_parts"]
            page_data.append(scraper(wd, row["link"], path_parts, timeout=30 if index == 0 else 5))
        except InvalidSessionIdException:
            wd = webdriver.Chrome(options=chrome_options)
            page_data.append(scraper(wd, row["link"], path_parts, timeout=30))
        except StaleElementReferenceException:
            continue
        except Exception as e:
            print(f"Error scraping {row['link']}: {e}")
            continue
    
    wd.quit()
    
    pprint(page_data[:5])  # Print first 5 for preview
    
    # Save to JSON
    output_file = f"data/{page_type}_meta_data.json"
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(page_data, json_file, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(page_data)} pages to {output_file}")

