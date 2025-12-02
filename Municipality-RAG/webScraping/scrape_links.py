"""
Extract links to scrape from municipality websites.
Supports Haifa and Tel Aviv municipalities.
"""

import os
import pprint
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urlparse, urljoin

load_dotenv()

# Municipality base URLs
MUNICIPALITY_BASE_URLS = {
    "haifa": "https://www.haifa.muni.il",
    "tel-aviv": "https://www.tel-aviv.gov.il"
}

def extract_path_parts(url, base_url):
    """Extract path parts from a URL relative to base URL."""
    parsed_url = urlparse(url)
    parsed_base = urlparse(base_url)
    
    if parsed_url.netloc != parsed_base.netloc:
        return []
    
    path = parsed_url.path
    parts = path.strip("/").split("/")
    return parts

def is_valid_url(url, base_url, subcategory=""):
    """Check if URL is valid for scraping."""
    parsed_url = urlparse(url)
    parsed_base = urlparse(base_url)
    
    # Must be from same domain
    if parsed_url.netloc != parsed_base.netloc:
        return False
    
    # If subcategory specified, must contain it
    if subcategory:
        path = parsed_url.path.lower()
        return subcategory.lower() in path
    
    return True

def get_sitemap_urls(sitemap_url):
    """Extract URLs from sitemap."""
    try:
        response = requests.get(sitemap_url, timeout=30)
        soup = BeautifulSoup(response.text, "xml")
        url_list = soup.find_all("loc")
        return [url.text for url in url_list]
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

def scrape_municipality_links(municipality="haifa", subcategory=""):
    """
    Scrape links from municipality website.
    
    Args:
        municipality: "haifa" or "tel-aviv"
        subcategory: Optional subcategory filter (e.g., "services", "news")
    
    Returns:
        List of URLs
    """
    base_url = MUNICIPALITY_BASE_URLS.get(municipality)
    if not base_url:
        print(f"Unknown municipality: {municipality}")
        return []
    
    # Try to get sitemap first
    sitemap_url = f"{base_url}/sitemap.xml"
    urls = get_sitemap_urls(sitemap_url)
    
    if not urls:
        # Fallback: try to scrape from homepage
        try:
            response = requests.get(base_url, timeout=30)
            soup = BeautifulSoup(response.text, "html.parser")
            links = soup.find_all("a", href=True)
            urls = []
            for link in links:
                href = link.get("href")
                if href:
                    full_url = urljoin(base_url, href)
                    if is_valid_url(full_url, base_url, subcategory):
                        urls.append(full_url)
        except Exception as e:
            print(f"Error scraping homepage: {e}")
            return []
    
    # Filter URLs
    filtered_urls = [url for url in urls if is_valid_url(url, base_url, subcategory)]
    
    return filtered_urls

if __name__ == "__main__":
    # Example usage
    municipality = "haifa"  # or "tel-aviv"
    subcategory = ""  # e.g., "services", "news", "announcements"
    
    filtered_urls = scrape_municipality_links(municipality, subcategory)
    pprint.pprint(filtered_urls[:20])  # Print first 20
    
    # Save to CSV
    df = pd.DataFrame({
        "link": filtered_urls,
        "path_parts": [extract_path_parts(url, MUNICIPALITY_BASE_URLS[municipality]) for url in filtered_urls]
    })
    
    output_file = f"data/{municipality}_{subcategory if subcategory else 'all'}_links.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(df)} links to {output_file}")

