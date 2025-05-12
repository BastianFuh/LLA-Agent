import logging
from typing import List

import ollama
import requests

from util import const

logger = logging.getLogger(__name__)


_capabilites_cache = dict()

_active_llm_models = const.OPTION_MODEL
_active_embedding_models = const.OPTION_EMBEDDING


def _ollama_get_capabilities(model: str) -> List[str]:
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


def _ollama_is_function_calling(model: str) -> bool:
    return "tools" in _ollama_get_capabilities(model)


def _ollama_is_embedding(model: str) -> bool:
    return "embedding" in _ollama_get_capabilities(model)


def _ollama_add_llm_model(models: const.OptionType, model):
    """Add a model to the list of models.
    If the model is a function calling model, it is added to the list of models.

    Args:
        models (const.OptionType): The list of models.
        model (str): The model name.
    """
    if _ollama_is_function_calling(model):
        name = f"Ollama: {model}"
        models[name] = (model, const.PROVIDER_OLLAMA)
        logger.info(f"Added model {name} to OPTION_MODEL")


def _ollama_add_embedding_model(models: const.OptionType, model):
    """Add a model to the list of models.
    If the model is an embedding model, it is added to the list of models.

    Args:
        models (const.OptionType): The list of models.
        model (str): The model name.
    """
    if _ollama_is_embedding(model):
        name = f"Ollama: {model}"
        models[name] = (model, const.PROVIDER_OLLAMA)
        logger.info(f"Added model {name} to OPTION_EMBEDDING")


def init_models():
    """Initialize the list of LLM models.

    This function is called when the application starts to populate the list of models.

    Returns:
        const.OptionType: The list of LLM models.
    """
    global _active_llm_models, _active_embedding_models

    for model in ollama.list().models:
        _ollama_add_llm_model(_active_llm_models, model["model"])
        _ollama_add_embedding_model(_active_embedding_models, model["model"])


def get_llm_models() -> const.OptionType:
    """Get the list of LLM models.

    Returns:
        const.OptionType: The list of LLM models.
    """
    global _active_llm_models

    return _active_llm_models


def get_embedding_models() -> const.OptionType:
    """Get the list of embedding models.

    Returns:
        const.OptionType: The list of embedding models.
    """
    global _active_embedding_models

    return _active_embedding_models
