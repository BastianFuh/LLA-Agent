import asyncio
import json
import logging
import os
import traceback
from typing import Any, Dict, List, Optional

import aiohttp
import html2text
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.tools.google import GoogleSearchToolSpec
from llama_index.tools.tavily_research import TavilyToolSpec


async def fetch_webpage(
    url: str, session: aiohttp.ClientSession, timeout: int = 10
) -> Dict[str, str]:
    """Fetch a webpage using aiohttp and return its content as a string.

    Args:
        url (str): The URL of the webpage to fetch.
        timeout (int, optional): The timeout for the request in seconds. Defaults to 10.

    Returns:
        str: The content of the webpage as a string.
    """
    try:
        async with session.get(url, timeout=timeout) as response:
            return {"text": await response.text(), "url": str(response.url)}

    except asyncio.TimeoutError or aiohttp.ClientError as e:
        logging.error(f"Error fetching {url}: {e}")
        return None


async def fetch_webpages(
    urls: List[str],
    session: aiohttp.ClientSession,
) -> List[Document]:
    """Fetch multiple webpages concurrently using aiohttp."""

    documents = list()

    tasks = [asyncio.ensure_future(fetch_webpage(url, session)) for url in urls]
    responses = await asyncio.gather(*tasks)

    documents.extend(
        [
            Document(text=html2text.html2text(response["text"]), id_=response["url"])
            for response in responses
            if response is not None
        ]
    )

    return documents


async def summarize_website(
    url: str,
    query: str,
    session: Any = None,  # Technically this is a aiohttp.ClientSession, but this creates problems with pydantic
) -> str:
    """
    Find information about a query on a specified website. Returns a summary of
    the desired information.

    Args:
        url (str): The url of a website which information should be summarized
        query (str): The query that determines what information should be summarized

    Returns:
        str: A summary of the information on a website in regards to a specified query.
    """
    logging.info(f"Called summarize_website for {url} with {query}")

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    Settings.llm = OpenAI(model="gpt-4o-mini-2024-07-18")

    if session is not None:
        documents = await fetch_webpages([url], session)
    else:
        async with aiohttp.ClientSession() as session:
            documents = await fetch_webpages([url], session)

    if len(documents) == 0:
        logging.debug(f"Error {url} could not be accessed")
        return f"There was an error with summarizing {url}. The website could not be accessed."

    index = VectorStoreIndex.from_documents(documents)

    input_query = f"Summarize in very meticulously detail. {query}"

    return f"Source: {url} \n Content:{index.as_query_engine().query(input_query).response}"


async def summarize_websites(urls: list[str], query: str) -> list[str]:
    """
    Find information about a query on set of specified websites. Returns a summary per website of
    the desired information.

    This function is more efficient than calling summarize_website for each website separately.
    It uses the asyncio library to fetch the webpages concurrently. Therefore, it is recommended to
    use this function instead of summarize_website when multiple websites should be summarized.

    Args:
        urls (list[str]): A list of url of websites which information should be summarized
        query (str): The query that determines what information should be summarized

    Returns:
        list[str]: A list of summaries of the information on a set of websites in regards to a specified query.
    """
    logging.info(f"Called summarize_website for {len(urls)} urls with {query}")
    summaries = list()

    async with aiohttp.ClientSession() as session:
        request = [
            asyncio.ensure_future(summarize_website(url, query, session))
            for url in urls
        ]
        summaries.append(await asyncio.gather(*request))

    return summaries


def google_websearch(query: str, max_results: Optional[int] = 6) -> List[dict]:
    """
    Make a query to the Google search engine to receive a list of results. The query can use
    the google search operators which allows the composition of a sophisticated query.

    For the best performance the query should follow these points:
    - Be precise
    - Use relevant keywords
    - Use the google search operators for complex searches

    Args:
        query (str): The query to be passed to google search.
        max_results (int, optional): The number of search results to return. Defaults to 6.

    Returns:
        results: A list of dictionaries containing the results:
            url: The url of the result.
            content: The content of the result.
    """
    try:
        logging.info(f"Called google_search for {query}")
        search_engine = GoogleSearchToolSpec(
            key=os.getenv("GOOGLE_API_KEY"),
            engine=os.getenv("GOOGLE_SEARCH_ENGINE"),
            num=max_results,
        )

        search_results = search_engine.google_search(query)

        result = list()
        for document in search_results:
            json_data = json.loads(document.text)

            if json_data.__contains__("items"):
                for search_result in json_data["items"]:
                    if search_result.__contains__("snippet"):
                        result.append(
                            {
                                "url": search_result["link"],
                                "content": f"Title: {search_result['title']} Content \n {search_result['snippet']}",
                            }
                        )
    except Exception as e:
        print(e)
        print(traceback.format_exc())

    return result


def tavily_search(query: str, max_results: Optional[int] = 6) -> List[dict]:
    """
    Run query through Tavily Search and return metadata.

    Args:
        query: The query to search for.
        max_results: The maximum number of results to return.

    Returns:
        results: A list of dictionaries containing the results:
            url: The url of the result.
            content: The content of the result.

    """
    logging.info(f"Called tavily_search for {query}")
    search_engine = TavilyToolSpec(None)

    search_results = search_engine.search(query, max_results)

    results = list()
    for document in search_results:
        results.append(
            {
                "url": document.metadata["url"],
                "content": f"Content {document.text}",
            }
        )

    return results


# This functions should prevent models from repeating their last action because they think
# they need to some further steps which results in them calling the last action again. This
# happens because they are instructred to either give an answer or do a toolcall.
# Although it might also be extended to actually do more work, i.e., keep track of thoughts and
# remind the model of them.
def think(thought: str) -> str:
    """A function for assiting in thinking about complex problems. It receives informations about
    the current thought process which should be remembered.

    Args:
        tought (str): Information about the current though process which should be remembered.

    Returns:
        str: Information about the current though process.
    """
    return f"Your are currently thinking about {thought}"
