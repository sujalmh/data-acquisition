# Agent API

This project is a **FastAPI-based AI agent** that integrates **LangChain**, **Playwright**, and **various tools** to facilitate automated web crawling, Google searches, and File relevance checking.

## Features
- **Web Crawler**: Extracts clickable elements, API links, and file links from a given URL.
- **Google Search**: Performs site-restricted Google searches.
- **File Relevance Checker**: Downloads a File and extracts the first few lines to determine its relevance.
- **AI Agent**: Uses LangChain with GPT-4o-mini for query handling.

## Installation
### Prerequisites
- Python 3.12+
- [Playwright](https://playwright.dev/python/docs/intro) installed and browsers set up:
  ```sh
  playwright install
  ```

### Steps
1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Run the API:
   ```sh
   uvicorn main:app --host 127.0.0.1 --port 8000 --reload
   ```

## API Endpoints
- `GET /` - Health check.
- `POST /query` - Accepts a JSON payload with a query and returns AI-generated results.
  ```json
  {
    "query": "Collect file or api links for India CPI data 2025."
  }
  ```
    ```json
  {
    "query": "Collect file or api links for India Monetary Policy Report."
  }
  ```
