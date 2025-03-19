import sys
import asyncio

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import os
import json
import requests
from io import BytesIO

import pymupdf4llm
from googlesearch import search
from playwright.async_api import async_playwright

from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.chat_models import ChatOpenAI

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ensure the output folder exists
OUTPUT_FOLDER = "OUTPUT_FOLDER"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def google_search_tool(input_str):
    search_query = input_str
    try:
        # Support query parsing as "query | base_url"
        query, base_url = input_str.split('|', 1)
        query = query.strip()
        base_url = base_url.strip()
        search_query = f"site:{base_url} {query}"
    except Exception:
        pass
    try:
        results = list(search(search_query, num_results=10))
        return "\n".join(results)
    except Exception as e:
        return f"Error: {e}"

async def crawl_website(input_str):
    """
    Asynchronously crawls a URL or clicks an element using a selector,
    returning all clickable elements, API links, and file links.
    It extracts context from buttons and anchor tags by getting text from all child elements.
    If any clickable element's text contains 'api' or 'file', its link is added to the corresponding list.
    Returns CSS selectors for each clickable element.
    """
    try:
        # Check if input_str is already a dict; if not, try to parse it as JSON.
        if isinstance(input_str, dict):
            input_data = input_str
        else:
            input_data = json.loads(input_str)
        url = input_data["url"]
        click_selector = input_data.get("click_selector", None)
    except Exception:
        return "Invalid input. Expecting JSON with 'url' and optionally 'click_selector'."

    if url.endswith(".pdf") or url.endswith(".csv"):
        return "Cannot crawl PDF or CSV files. Use the 'File Relevance Checker' tool instead."

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        try:
            await page.goto(url)
            await page.wait_for_load_state('networkidle')
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(2000)
        except Exception as e:
            await browser.close()
            return f'Invalid URL. Error: {str(e)}'
        
        if click_selector:
            try:
                element = page.locator(click_selector)
                if element:
                    await element.click()
                    await page.wait_for_load_state('networkidle')
                    await page.wait_for_timeout(1000)
                else:
                    raise Exception("No element found for selector")
            except Exception as e:
                await browser.close()
                return f'Error clicking element: {str(e)}'

        elements = await page.query_selector_all("a, button")
        clickable_elements = []
        api_links, file_links = [], []
        file_extensions = (".pdf", ".xls", ".xlsx", ".csv", ".json")

        for idx, el in enumerate(elements):
            tag = await el.evaluate("el => el.tagName.toLowerCase()")
            text = (await el.inner_text()).strip()
            attributes = await el.evaluate(
                "el => [...el.attributes].map(attr => `${attr.name}='${attr.value}'`).join(' ')"
            )
            href = await el.get_attribute("href") if tag == "a" else None
            css_selector = await el.evaluate(
                "el => el.tagName.toLowerCase() + "
                "(el.id ? '#' + el.id : '') + "
                "(el.className ? '.' + el.className.split(' ').join('.') : '')"
            )

            clickable_elements.append({
                "index": idx,
                "tag": tag,
                "text": text,
                "attributes": attributes,
                "href": href,
                "css_selector": css_selector
            })

            text_lower = text.lower()
            if "api" in text_lower and href and href not in api_links:
                api_links.append(href)
            if "file" in text_lower and href and href not in file_links:
                file_links.append(href)
            if href:
                href_lower = href.lower()
                if "/api/" in href_lower and href not in api_links:
                    api_links.append(href)
                if href_lower.endswith(file_extensions) and href not in file_links:
                    file_links.append(href)

        current_url = page.url
        await browser.close()

        return {
            "current_url": current_url,
            "clickable_elements": clickable_elements,
            "api_links": api_links,
            "file_links": file_links
        }

def sync_crawl_website(input_str):
    """
    Synchronous wrapper for the asynchronous crawl_website function.
    """
    result = asyncio.run(crawl_website(input_str))
    return json.dumps(result)

def check_file_relevance(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return ""
        content_type = response.headers.get('Content-Type', '').lower()
        file_content = BytesIO(response.content)
        filename = url.rsplit("/", 1)[-1].lower()

        if 'application/pdf' in content_type or filename.endswith('.pdf'):
            with open(f"{OUTPUT_FOLDER}/{filename}", "wb") as f:
                f.write(file_content.getbuffer())
            page_data = pymupdf4llm.to_markdown(f"{OUTPUT_FOLDER}/{filename}", pages=[0])
            return page_data[:1000].lower()
        
        elif 'text/csv' in content_type or filename.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(file_content, nrows=5)
            return df.to_string().lower()
        
        else:
            return ""
    except Exception as e:
        print(f"Error checking file {url}: {e}")
        return ""

# Define LangChain agent and tools
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)

tools = [
    Tool(
        name="Web Crawler",
        func=sync_crawl_website,
        description=(
            "Crawls a URL or clicks an element using a CSS or XPath selector, extracting clickable elements, "
            "API links, and file links. Input is a JSON string with 'url' (required) and optionally 'click_selector'. "
            "Returns a JSON string with 'current_url', 'clickable_elements', 'api_links', and 'file_links'."
        )
    ),
    Tool(
        name="Google Search",
        func=google_search_tool,
        description=(
            "Performs a Google search with the query restricted to the base URL. Input should be 'query' or "
            "'query site:url'. Returns a list of URLs."
        )
    ),
    Tool(
        name="File Relevance Checker",
        func=check_file_relevance,
        description=(
            "Checks if a File is relevant by examining the first few lines. Input is strictly a File URL."
        )
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

app = FastAPI(
    title="Agent API",
    description="A FastAPI wrapping a LangChain Agent",
    version="1.0"
)

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def run_query(request: QueryRequest):
    try:
        response = agent.run(request.query)
        print(response)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {
        "message": "Agent API is running. Use the /query endpoint to interact with the agent."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
