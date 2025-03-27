from typing import List, Optional
from llama_index.tools.google import GoogleSearchToolSpec
from llama_index.tools.tavily_research import TavilyToolSpec

import os
import json


def google_search(query: str, max_results: Optional[int] = 6) -> List[dict]:
    """
    Make a query to the Google search engine to receive a list of results.

    Args:
        query (str): The query to be passed to Google search.
        max_results (int, optional): The number of search results to return. Defaults to None.

    Raises:
        ValueError: If the 'num' is not an integer between 1 and 10.

    Returns:
        results: A list of dictionaries containing the results:
            url: The url of the result.
            content: The content of the result.
    """
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
