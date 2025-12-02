"""
City Scraper for Israeli Municipalities
Scrapes municipal information from Haifa and Tel Aviv websites.
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import logging

logger = logging.getLogger(__name__)

# Municipality website URLs
MUNICIPALITY_URLS = {
    "חיפה": "https://www.haifa.muni.il",
    "תל אביב": "https://www.tel-aviv.gov.il",
    "תל-אביב": "https://www.tel-aviv.gov.il",
}

def get_city_info(city_name: str) -> list:
    """
    Scrape municipal information for a given city.
    
    Args:
        city_name: Hebrew name of the city (e.g., "חיפה", "תל אביב")
    
    Returns:
        List of information strings about municipal services, announcements, etc.
    """
    # Normalize city name
    city_name = city_name.strip()
    
    # Check if city is supported
    if city_name not in MUNICIPALITY_URLS:
        logger.warning(f"City {city_name} not supported. Supported cities: {list(MUNICIPALITY_URLS.keys())}")
        return [f"מידע על {city_name} אינו זמין כרגע. הערים הנתמכות הן חיפה ותל אביב."]
    
    url = MUNICIPALITY_URLS[city_name]
    
    try:
        # Setup Chrome options for headless mode
        opts = webdriver.ChromeOptions()
        opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.binary_location = "/usr/bin/chromium"
        
        service = Service("/usr/bin/chromedriver")
        driver = webdriver.Chrome(service=service, options=opts)
        
        driver.get(url)
        
        info_items = []
        
        if city_name == "חיפה" or city_name == "Haifa":
            info_items = scrape_haifa(driver)
        elif city_name == "תל אביב" or city_name == "תל-אביב" or city_name == "Tel Aviv":
            info_items = scrape_tel_aviv(driver)
        
        driver.quit()
        return info_items if info_items else [f"לא נמצא מידע זמין עבור {city_name} כרגע."]
        
    except TimeoutException:
        logger.error(f"Timeout while scraping {city_name}")
        return [f"זמן המתנה פג בעת ניסיון לקבל מידע על {city_name}. נסה שוב מאוחר יותר."]
    except Exception as e:
        logger.error(f"Error scraping {city_name}: {str(e)}")
        return [f"שגיאה בעת קבלת מידע על {city_name}: {str(e)}"]


def scrape_haifa(driver: webdriver.Chrome) -> list:
    """
    Scrape information from Haifa municipality website.
    
    Args:
        driver: Selenium WebDriver instance
    
    Returns:
        List of information strings
    """
    info_items = []
    
    try:
        # Wait for page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Try to find announcements or news sections
        # This is a generic implementation - adjust selectors based on actual website structure
        try:
            # Look for common announcement/news elements
            announcements = driver.find_elements(By.CSS_SELECTOR, ".announcement, .news-item, .alert, [class*='announcement'], [class*='news']")
            for ann in announcements[:10]:  # Limit to first 10
                text = ann.text.strip()
                if text:
                    info_items.append(f"הודעה: {text}")
        except NoSuchElementException:
            pass
        
        # Try to find service links or information
        try:
            services = driver.find_elements(By.CSS_SELECTOR, "a[href*='service'], a[href*='service'], .service-link")
            for service in services[:10]:  # Limit to first 10
                text = service.text.strip()
                if text:
                    info_items.append(f"שירות: {text}")
        except NoSuchElementException:
            pass
        
        # If no specific elements found, get general page info
        if not info_items:
            try:
                main_content = driver.find_element(By.TAG_NAME, "main")
                paragraphs = main_content.find_elements(By.TAG_NAME, "p")[:5]
                for p in paragraphs:
                    text = p.text.strip()
                    if text and len(text) > 20:
                        info_items.append(text)
            except NoSuchElementException:
                pass
        
    except Exception as e:
        logger.error(f"Error in scrape_haifa: {str(e)}")
    
    return info_items


def scrape_tel_aviv(driver: webdriver.Chrome) -> list:
    """
    Scrape information from Tel Aviv municipality website.
    
    Args:
        driver: Selenium WebDriver instance
    
    Returns:
        List of information strings
    """
    info_items = []
    
    try:
        # Wait for page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Try to find announcements or news sections
        # This is a generic implementation - adjust selectors based on actual website structure
        try:
            # Look for common announcement/news elements
            announcements = driver.find_elements(By.CSS_SELECTOR, ".announcement, .news-item, .alert, [class*='announcement'], [class*='news']")
            for ann in announcements[:10]:  # Limit to first 10
                text = ann.text.strip()
                if text:
                    info_items.append(f"הודעה: {text}")
        except NoSuchElementException:
            pass
        
        # Try to find service links or information
        try:
            services = driver.find_elements(By.CSS_SELECTOR, "a[href*='service'], a[href*='service'], .service-link")
            for service in services[:10]:  # Limit to first 10
                text = service.text.strip()
                if text:
                    info_items.append(f"שירות: {text}")
        except NoSuchElementException:
            pass
        
        # If no specific elements found, get general page info
        if not info_items:
            try:
                main_content = driver.find_element(By.TAG_NAME, "main")
                paragraphs = main_content.find_elements(By.TAG_NAME, "p")[:5]
                for p in paragraphs:
                    text = p.text.strip()
                    if text and len(text) > 20:
                        info_items.append(text)
            except NoSuchElementException:
                pass
        
    except Exception as e:
        logger.error(f"Error in scrape_tel_aviv: {str(e)}")
    
    return info_items


# For testing purposes (commented out for production)
# if __name__ == "__main__":
#     import sys
#     city = sys.argv[1] if len(sys.argv) > 1 else "חיפה"
#     for line in get_city_info(city):
#         print(line)

