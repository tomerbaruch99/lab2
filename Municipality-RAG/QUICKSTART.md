# Quick Start Guide

## Prerequisites Checklist

- [ ] Python 3.8+ installed
- [ ] Docker installed (for scraping service)
- [ ] Pinecone account and API key
- [ ] Google AI API key

## Step-by-Step Setup

### 1. Environment Setup

Copy the example environment file:
```bash
cp env.example .env
```

Edit `.env` and add your API keys:
- `PINECONE_API_KEY`
- `PINECONE_INDEX` (create an index in Pinecone first)
- `PINECONE_ENVIRONMENT`
- `GOOGLE_API_KEY`

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Scrape Municipal Data (Optional)

If you want to scrape data yourself:

```bash
# Scrape links from municipality websites
cd webScraping
python scrape_links.py

# Scrape individual pages
python scrape_page.py
```

### 4. Index Data to Pinecone

```bash
# Run the Jupyter notebook
jupyter notebook upload_to_Pinecone.ipynb
```

Execute all cells to upload your scraped data to Pinecone.

### 5. Start the Scraping API Service

```bash
cd municipalities_scraping
docker build -t municipality-api .
docker run -d -p 8081:8081 --name municipality-api municipality-api
```

Verify it's running:
```bash
curl http://localhost:8081/health
```

### 6. Start the Main Application

```bash
streamlit run Chatbot_logic.py
```

The application will open at `http://localhost:8501`

## Testing

### Test the API

```bash
curl -X POST http://localhost:8081/get_city_info \
  -H "Content-Type: application/json" \
  -d '{"city_name": "חיפה"}'
```

### Test the Chatbot

1. Open `http://localhost:8501` in your browser
2. Try asking questions in Hebrew:
   - "מה השירותים העירוניים הזמינים?"
   - "איך משלמים ארנונה?"
   - "מה שעות פעילות העירייה?"

## Troubleshooting

### API Service Not Starting

- Check Docker is running: `docker ps`
- Check logs: `docker logs municipality-api`
- Verify port 8081 is not in use

### Pinecone Connection Issues

- Verify your API key and index name in `.env`
- Check your Pinecone index exists and has the correct dimensions
- Ensure your Pinecone environment matches your account

### Chatbot Not Responding

- Verify the API service is running on port 8081
- Check browser console for errors
- Verify all environment variables are set correctly

## Next Steps

- Customize the UI in `UI.py`
- Adjust namespace classification in `Chatbot_logic.py`
- Add more cities by updating `City_Scraper.py`
- Enhance scraping logic in `webScraping/` files

