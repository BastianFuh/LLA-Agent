import grequests

import os
import json

import logging

import traceback
import html2text

from typing import List, Optional
from llama_index.tools.google import GoogleSearchToolSpec
from llama_index.tools.tavily_research import TavilyToolSpec
from llama_index.core import VectorStoreIndex

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import Document


def load_webpages(urls: List[str]):
    documents = list()

    request = [grequests.get(url, timeout=10) for url in urls]

    responses = grequests.map(requests=request, size=4)

    documents.extend(
        [
            Document(text=html2text.html2text(response.text), id_=response.url)
            for response in responses
            if response is not None
        ]
    )

    return documents


def summarize_website(url: str, query: str) -> str:
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

    documents = load_webpages([url])

    if len(documents) == 0:
        logging.debug(f"Error {url} could not be accessed")
        return f"There was an error with summarizing {url}. The website could not be accessed."

    index = VectorStoreIndex.from_documents(documents)

    input_query = f"Summarize in very meticulously detail. {query}"

    return f"Source: {url} \n Content:{index.as_query_engine().query(input_query).response}"


def summarize_websites(urls: list[str], query: str) -> list[str]:
    """
    Find information about a query on set of specified websites. Returns a summary per website of
    the desired information.

    Args:
        urls (list[str]): A list of url of websites which information should be summarized
        query (str): The query that determines what information should be summarized

    Returns:
        list[str]: A list of summaries of the information on a set of websites in regards to a specified query.
    """
    logging.info(f"Called summarize_website for {len(urls)} urls with {query}")
    summaries = list()
    for url in urls:
        summaries.append(summarize_website(url, query))

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
            for search_result in json.loads(document.text)["items"]:
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
