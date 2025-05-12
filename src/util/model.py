import asyncio
import logging
from typing import List

import ollama
import requests

from util import const

logger = logging.getLogger(__name__)


_capabilites_cache = dict()


async def _ollama_get_capabilities(model: str) -> List[str]:
    """Get the capabilities of the model.

    This needs to be done because the ollama package does not provide a way to get the capabilities of the model.

    Args:
        model (str): The model name.

    Returns:
        list[str]: The capabilities of the model.
    """
    global _capabilites_cache

    if _capabilites_cache.__contains__(model):
        return _capabilites_cache[model]

    response = requests.post("http://localhost:11434/api/show", json={"model": model})

    if response.status_code == 200:
        _capabilites_cache[model] = response.json()["capabilities"]
        return response.json()["capabilities"]

    return []


async def _ollama_is_function_calling(model: str) -> bool:
    return "tools" in await _ollama_get_capabilities(model)


async def _ollama_is_embedding(model: str) -> bool:
    return "embedding" in await _ollama_get_capabilities(model)


async def _ollama_add_llm_model(models: const.OptionType, model):
    """Add a model to the list of models.

    Args:
        models (const.OptionType): The list of models.
        model (str): The model name.
    """
    if await _ollama_is_function_calling(model):
        name = f"Ollama: {model}"
        models[name] = (model, const.PROVIDER_OLLAMA)
        logger.info(f"Added model {name} to OPTION_MODEL")


async def _ollama_add_embedding_model(models: const.OptionType, model):
    """Add a model to the list of models.

    Args:
        models (const.OptionType): The list of models.
        model (str): The model name.
    """
    if await _ollama_is_embedding(model):
        name = f"Ollama: {model}"
        models[name] = (model, const.PROVIDER_OLLAMA)
        logger.info(f"Added model {name} to OPTION_EMBEDDING")


# for ollama_model in ollama.list().models:
#     model = ollama_model["model"]
#     if ollama_is_function_calling(model):
#         name = f"Ollama: {model}"
#         OPTION_MODEL[name] = (model, PROVIDER_OLLAMA)
#         logger.info(f"Added model {name} to OPTION_MODEL")
#
#     if ollama_is_embedding(model):
#         name = f"Ollama: {model}"
#         OPTION_EMBEDDING.append((name, model))
#         logger.info(f"Added model {name} to OPTION_EMBEDDING")


def get_llm_models() -> const.OptionType:
    """Get the list of LLM models.

    Returns:
        const.OptionType: The list of LLM models.
    """

    base_models = const.OPTION_MODEL

    async def exec():
        ollama_models = ollama.list().models

        model_requests = [
            asyncio.ensure_future(_ollama_add_llm_model(base_models, model["model"]))
            for model in ollama_models
        ]
        await asyncio.gather(*model_requests)

    asyncio.run(exec())

    return base_models


def get_embedding_models() -> const.OptionType:
    """Get the list of embedding models.

    Returns:
        const.OptionType: The list of embedding models.
    """

    base_models = const.OPTION_EMBEDDING

    async def exec():
        ollama_models = ollama.list().models

        model_requests = [
            asyncio.ensure_future(
                _ollama_add_embedding_model(base_models, model["model"])
            )
            for model in ollama_models
        ]
        await asyncio.gather(*model_requests)

    asyncio.run(exec())

    return base_models
