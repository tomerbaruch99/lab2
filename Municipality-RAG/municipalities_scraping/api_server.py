from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from City_Scraper import get_city_info
import logging
import time
from typing import List
import sys
import os

# Force UTF-8 encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# Set environment variables for UTF-8
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['LANG'] = 'en_US.UTF-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'

# Configure logging with UTF-8
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Israeli Municipality Information API",
    description="Get real-time municipal information for Israeli cities (Haifa, Tel Aviv)",
    version="1.0.0"
)

class CityRequest(BaseModel):
    city_name: str
    
    class Config:
        # Ensure UTF-8 encoding for Pydantic
        str_strip_whitespace = True

class CityInfoResponse(BaseModel):
    city_name: str
    info: List[str]
    count: int
    timestamp: str

@app.get("/")
async def root():
    return {"message": "Israeli Municipality Information API", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.strftime('%H:%M:%S')}

@app.post("/get_city_info", response_model=CityInfoResponse)
async def get_municipality_info(request: CityRequest):
    """
    Get current municipal information and services for an Israeli city.
    
    - **city_name**: Hebrew name of the Israeli city (e.g., "חיפה" for Haifa, "תל אביב" for Tel Aviv)
    """
    try:
        city_name = request.city_name.strip()
        if not city_name:
            raise HTTPException(status_code=400, detail="city_name cannot be empty")
        
        # Ensure proper UTF-8 encoding
        if isinstance(city_name, str):
            city_name = city_name.encode('utf-8').decode('utf-8')
        
        logger.info(f"Getting municipal info for: {repr(city_name)}")  # Use repr to see actual characters
        
        # Call your scraper function
        info = get_city_info(city_name)
        
        return CityInfoResponse(
            city_name=city_name,
            info=info,
            count=len(info),
            timestamp=time.strftime('%H:%M:%S, %d/%m/%Y')
        )
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error getting municipal info for {repr(city_name)}: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Error retrieving municipal information: {error_msg}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)

