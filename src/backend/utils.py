from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.deepseek import DeepSeek
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.openrouter import OpenRouter

from backend import tools
from util import const
from util.model import get_llm_models


async def get_llms_tools(ctx: Context) -> list:
    llm_tools = list([FunctionTool.from_defaults(tools.think)])

    search_engine = await ctx.get(const.SEARCH_ENGINE, default=const.NONE)

    if search_engine == const.TAVILY:
        # API Keys are provided via the environment but because the tools do not have
        # default values "None" needs to be passed to them
        llm_tools.extend([FunctionTool.from_defaults(tools.tavily_search)])

    if search_engine == const.GOOGLE:
        llm_tools.extend([FunctionTool.from_defaults(tools.google_websearch)])

    if search_engine != const.NONE:
        llm_tools.extend(
            [
                FunctionTool.from_defaults(tools.summarize_website),
                FunctionTool.from_defaults(tools.summarize_websites),
            ]
        )

    return llm_tools


def get_llm(model: str):
    model_information = get_llm_models()[model]

    model_name = model_information[0]

    match model_information[1]:
        case const.PROVIDER_DEEPSEEK:
            return DeepSeek(
                model=model_name,
                is_function_calling_model=True,
                is_chat_model=True,
            )
        case const.PROVIDER_OPENAI:
            return OpenAI(model=model_name)
        case const.PROVIDER_OPENROUTER:
            return OpenRouter(model=model_name)
        case const.PROVIDER_OLLAMA:
            # Ollama does not like temperatures above 1
            return Ollama(
                model=model_name,
                is_chat_model=True,
            )
        case _:
            raise ValueError("Unknown Provider")
