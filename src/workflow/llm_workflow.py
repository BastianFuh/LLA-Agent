from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    Context,
    step,
)
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.base.llms.types import ChatMessage

from workflow.events import LLM_Progress_Event, LLM_StartEvent, LLMStopEvent


def build_message(message: str, history: list[dict]):
    llm_message = list()
    for element in history:
        llm_message.append(
            ChatMessage(role=element["role"], content=element["content"])
        )

    llm_message.append(ChatMessage(role="user", content=message))

    return llm_message


class LLM_FLow(Workflow):
    llm = OpenRouter(model="deepseek/deepseek-chat-v3-0324:free")

    @step
    async def start(self, ctx: Context, ev: LLM_StartEvent) -> LLMStopEvent:
        is_stream = await ctx.get("is_stream", default=False)

        if is_stream:
            gen = await self.llm.astream_chat(build_message(ev.message, ev.history))

            async for response in gen:
                ctx.write_event_to_stream(
                    LLM_Progress_Event(response_delta=response.delta)
                )

            return LLMStopEvent(result="")
        else:
            response = await self.llm.achat(build_message(ev.message, ev.history))

            return LLMStopEvent(result=response.message.content)
