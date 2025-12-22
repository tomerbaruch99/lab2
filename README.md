<div align="center">

# 🏛️ Haifa Municipality RAG Project

**Intelligent Question-Answering System for Municipal Services**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-red.svg)](https://streamlit.io/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-yellow.svg)](https://www.pinecone.io/)
[![Gemini](https://img.shields.io/badge/Gemini-LLM-green.svg)](https://ai.google.dev/)

<img src="logos/logo1.png" alt="Haifa Municipality Logo" width="200"/>

</div>

---

## Overview

A **RAG (Retrieval-Augmented Generation)** system that enables citizens to ask questions in Hebrew about Haifa municipal services, regulations, and information.

### 🗝️ Key Features

- 🇮🇱 **Hebrew Language Support** - Native support for Hebrew queries and responses
- 🎯 **Smart Retrieval** - Adapted chunking strategy
- 🧠 **Query Enhancement** - Automatic query expansion and reranking for better results
- 📊 **Confidence Scoring** - Indicators for answer reliability
- 🔍 **Smart Page Finder** - Suggests relevant official municipal pages
- 📈 **Comprehensive Evaluation** - Testing framework with baseline comparisons
- 🎨 **Modern Web UI** - Streamlit interface for easy interaction

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+**
- **Scraped data**: [Download haifa_scraped.json](https://technionmail-my.sharepoint.com/:u:/g/personal/amit_shirazi_campus_technion_ac_il/EcLo4Nc_EyBHmCe8jC5R8RsBDPihBFq3K_3LUQRGqRXrNA?e=cbqSSN)
- **API keys**: Pinecone and Gemini API keys

### Installation & Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt
# Optional: pip install scikit-learn>=1.0.0 for baseline evaluation methods

# 2. Configure API keys
# Create utils/api_keys.json:
# {
#   "PINECONE_API_KEY": "your-pinecone-key",
#   "GEMINI_API_KEY": "your-gemini-key"
# }

# 3. Prepare data
python scrape_and_prepare_data/data_preparation.py \
    --input_json scrape_and_prepare_data/haifa_scraped.json \
    --out_dir scrape_and_prepare_data/haifa_prepared_data

# 4. Index to Pinecone
python indexing.py \
    --prepared_file scrape_and_prepare_data/haifa_prepared_data/haifa_rag_chunks.parquet

# 5. (Optional) Build page index for Smart Page Finder
python utils/build_page_index.py

# 6. Launch application
streamlit run chatbot.py
```

---

## 💻 Usage

### Web Interface (Recommended)

```bash
streamlit run chatbot.py
```

**Features:**
- 💬 Natural language queries in Hebrew
- 🎯 Real-time answer generation
- 📊 Confidence indicators (🟢 High / 🟡 Medium / 🔴 Low)
- 🔗 Smart Page Finder suggestions
- 📝 Conversation history

### Command Line Interface

**Basic usage:**
```bash
python gemini_integration.py --question "איך משלמים ארנונה?" --top_k 5
```

**With query enhancement and reranking:**
```bash
python gemini_integration.py \
    --question "איך משלמים ארנונה?" \
    --top_k 5 \
    --use_query_enhancement \
    --use_reranking
```

### Code Examples

Explore practical usage examples in the `examples/` folder:

| Example | Description |
|---------|-------------|
| `example_retriever_usage.py` | Basic retrieval operations |
| `example_prompt_builder.py` | Custom prompt construction |
| `example_gemini_rag.py` | Complete RAG pipeline |
| `example_smart_page_finder.py` | Smart Page Finder integration |
| `example_retrieval_diagnostics.py` | Debugging retrieval issues |
| `example_chatbot_integration.py` | Chatbot integration patterns |

See [`examples/README.md`](examples/README.md) for detailed descriptions.

---

## 📊 Evaluation

Comprehensive evaluation framework for testing and comparing different configurations.

### Quick Start

```bash
# Generate evaluation results (requires API access)
python evaluation/generate_evaluation_results.py \
    --testset_file tests/embedding_testset.json \
    --output_dir evaluation/all_strategies_comparison_eval_results

# Analyze results (no API access needed)
jupyter notebook evaluation/analyze_evaluation_results.ipynb
```

See [`evaluation/README.md`](evaluation/README.md) and [`evaluation/EVALUATION_WORKFLOW.md`](evaluation/EVALUATION_WORKFLOW.md) for complete documentation.

---

## 🏗️ Architecture

### Core Components

| Component | Description |
|-----------|-------------|
| **🔍 Retriever** | Automatic namespace detection, metadata-aware retrieval, intelligent fallback search |
| **✍️ Prompt Builder** | Multiple prompt styles (detailed, concise, conversational, structured) |
| **🤖 Gemini Integration** | Complete RAG pipeline with query enhancement and reranking |
| **📊 Confidence Meter** | Real-time answer quality evaluation (🟢 High / 🟡 Medium / 🔴 Low) |
| **🔗 Smart Page Finder** | Intelligent suggestions for relevant official municipal pages |
| **📈 Evaluation System** | Comprehensive framework for comparing strategies, baselines, and configurations |

### Project Structure

```
project/
├── Core Components
│   ├── chatbot.py               # Streamlit web UI
│   ├── gemini_integration.py    # Complete RAG system
│   ├── retriever.py             # Semantic search & retrieval
│   ├── prompt_builder.py        # LLM prompt construction
│   ├── confidence_meter.py      # Answer quality scoring
│   └── indexing.py              # Pinecone indexing
│
├── scrape_and_prepare_data/     # Data scraping & chunking
├── evaluation/                   # Comprehensive testing framework
├── examples/                     # Usage examples & tutorials
└── utils/                        # Shared utilities & helpers
```

---

## 📖 Documentation

| Topic | Documentation |
|-------|---------------|
| 📚 **Examples** | [`examples/README.md`](examples/README.md) - Usage examples and code samples |
| 🧪 **Evaluation** | [`evaluation/README.md`](evaluation/README.md) - Evaluation framework |
| 🔄 **Evaluation Workflow** | [`evaluation/EVALUATION_WORKFLOW.md`](evaluation/EVALUATION_WORKFLOW.md) - Step-by-step guide |
| 📊 **Data Preparation** | [`scrape_and_prepare_data/README.md`](scrape_and_prepare_data/README.md) - Data scraping and preparation |
| 🛠️ **Utilities** | [`utils/README.md`](utils/README.md) - Utility functions |
| 🔗 **Smart Page Finder** | [`utils/SMART_PAGE_FINDER_README.md`](utils/SMART_PAGE_FINDER_README.md) - Page finder documentation |
| 🏗️ **Project Structure** | [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) - Complete file structure |

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Dimension mismatch error** | Delete and recreate the Pinecone index with the correct embedding model dimension |
| **No results returned** | Verify that data is indexed and the index name matches your configuration |
| **Low retrieval scores** | Check that embedding model matches, queries are in Hebrew, and data quality is good |
| **API key errors** | Ensure `utils/api_keys.json` exists with valid Pinecone and Gemini keys |
| **Import errors** | Run `pip install -r requirements.txt` to install all dependencies |

