from llama_index.llms.openrouter import OpenRouter
from llama_index.llms.openai import AsyncOpenAI, OpenAI

from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool


from util import const
from workflow import tools


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
    model_information = const.OPTION_MODEL[model]

    match model_information[1]:
        case const.PROVIDER_DEEPSEEK:
            raise NotImplementedError
        case const.PROVIDER_OPENAI:
            return OpenAI(model=model_information[0], temperature=1, max_tokens=1024)
        case const.PROVIDER_OPENROUTER:
            return OpenRouter(
                model=model_information[0], temperature=1, max_tokens=1024
            )
        case _:
            raise ValueError("Unknown Provider")
