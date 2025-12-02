# Municipality-RAG System 🏛️

A comprehensive Retrieval-Augmented Generation (RAG) system designed to provide real-time municipal information and assistance for Israeli municipalities. This system combines AI-powered chatbot capabilities with real-time data scraping to deliver accurate, up-to-date municipal guidance.

## 🔥 Features

- **AI-Powered Chatbot**: Intelligent conversational AI using Google's Gemini 2.5 Flash model
- **Real-Time Data**: Live scraping of municipal websites (Haifa and Tel Aviv)
- **Vector Database**: Pinecone-powered semantic search for accurate information retrieval
- **Multilingual Support**: Hebrew language support with RTL display
- **Streamlit UI**: User-friendly web interface with custom styling
- **Municipal APIs**: RESTful API for real-time city information
- **Conversation Memory**: Maintains chat history for contextual responses

## 🏗️ Architecture

The system consists of several key components:

1. **Main Application** (`Chatbot_logic.py`) - Streamlit-based web interface and chatbot logic
2. **UI** (`UI.py`) - Streamlit UI components
3. **API Tools** (`API_tools.py`) - Integration with external municipal APIs
4. **Municipalities Scraping** - Real-time data collection and processing
5. **Vector Store** - Pinecone database for semantic search
6. **Web Scraping** - Automated content extraction from municipal websites

## 📋 Prerequisites

- Python 3.8+
- Pinecone API Key
- Google AI API Key
- Docker (for the scraping service)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Municipality-RAG
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```.env
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX=your_index_name
   PINECONE_ENVIRONMENT=your_environment
   GOOGLE_API_KEY=your_google_api_key
   ```

4. **Initialize the vector database**
   ```bash
   jupyter notebook upload_to_Pinecone.ipynb
   ```

## 🏃‍♂️ Running the Application

### Launch Docker Container of real-time Tool
```bash
cd municipalities_scraping
docker build -t municipality-api .
docker run -d -p 8081:8081 --name municipality-api municipality-api
```

### Start the Application
```bash
streamlit run Chatbot_logic.py
```

The application will be available at `http://localhost:8501`

## 📁 Project Structure

```
Municipality-RAG/
├── Chatbot_logic.py              # Main script - defines the chatbot, all relevant LLM agents and runs the app
├── UI.py                         # Defines the UI of the web-page
├── API_tools.py                  # External API integrations
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables (you should create this)
├── upload_to_Pinecone.ipynb      # Vector database setup
├── data/                         # Scraped municipal data
│   ├── article_meta_data.json
│   ├── service_meta_data.json
│   └── announcement_meta_data.json
├── municipalities_scraping/      # Real-time data scraping
│   ├── api_server.py               # FastAPI scraping service
│   ├── City_Scraper.py              # City information scraper
│   ├── Dockerfile                   # Docker configuration
│   └── requirements.txt             # Scraping service dependencies
├── webScraping/                  # Data scraping scripts
│   ├── scrape_links.py             # Extraction of web-pages to scrape
│   └── scrape_page.py              # Scraping of web-pages
├── evaluation/                   # Evaluation notebooks and data
│   ├── evaluation.ipynb
│   └── data.csv
└── static/                       # Static assets
    └── logo.png
```

## 🔧 Configuration

### Vector Database Setup
The system uses Pinecone for vector storage. Initialize your database using the provided Jupyter notebook:
```bash
jupyter notebook upload_to_Pinecone.ipynb
```

### Supported Cities
Currently supports:
- **Haifa** (חיפה)
- **Tel Aviv** (תל אביב)

Future expansion planned for:
- **Be'er-sheva** (באר שבע)

## 💬 Usage

1. **Start the application** following the installation steps
2. **Ask questions** about municipal services, procedures, or information
3. **Get real-time updates** on city-specific information and services
4. **Receive AI-powered recommendations** based on your query

### Example Queries
- "מה השירותים העירוניים הזמינים?" (What municipal services are available?)
- "איך משלמים ארנונה?" (How do I pay property tax?)
- "מה שעות פעילות העירייה?" (What are the municipality's operating hours?)
- "איך מקבלים רישיון עסק?" (How do I get a business license?)

## 🔍 API Endpoints

### Scraping Service (`localhost:8081`)
- `POST /get_city_info` - Get real-time municipal information
  ```json
  {
    "city_name": "חיפה"
  }
  ```

## 📊 Evaluation

Our evaluation and result analysis can be found in [evaluation/evaluation.ipynb](evaluation/evaluation.ipynb),
or accessed by running:
```bash
jupyter notebook evaluation/evaluation.ipynb
```

Our evaluation dataset can be found in [evaluation/data.csv](evaluation/data.csv)

## 🛠️ Technologies Used

- **AI/ML**: LangChain, Google Gemini, Sentence Transformers
- **Vector Database**: Pinecone
- **Web Framework**: Streamlit, FastAPI
- **Data Processing**: Pandas, NumPy
- **Web Scraping**: Selenium
- **Evaluation**: Custom dataset, Tailored metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📝 License

This project is developed for educational purposes as part of a Data Analysis and Presentation Laboratory course.

## 🆘 Support

For issues and questions:
1. Check the existing documentation
2. Review the Jupyter notebooks for examples
3. Ensure all environment variables are properly set

## 🔮 Future Enhancements

- Support for additional municipalities (Be'er-sheva)
- Enhanced scraping capabilities
- Improved namespace classification
- Advanced evaluation metrics
- Multi-language support (English)

