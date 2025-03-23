from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    Context,
    step,
)
from llama_index.llms.openai import OpenAI
from llama_index.core.base.llms.types import ChatMessage

from workflow.events import LLM_Progress_Event


def build_message(message: str, history: list[dict]):
    llm_message = list()
    for element in history:
        llm_message.append(
            ChatMessage(role=element["role"], content=element["content"])
        )

    llm_message.append(ChatMessage(role="user", content=message))

    return llm_message


class LLM_FLow(Workflow):
    llm = OpenAI(model="gpt-4o-mini")

    @step
    async def start(self, ctx: Context, ev: StartEvent) -> StopEvent:
        is_stream = await ctx.get("is_stream", default=False)

        if is_stream:
            gen = await self.llm.astream_chat(build_message(ev.message, ev.history))

            async for response in gen:
                ctx.write_event_to_stream(
                    LLM_Progress_Event(response_delta=response.delta)
                )

            return StopEvent(result="")
        else:
            response = await self.llm.achat(build_message(ev.message, ev.history))

            return StopEvent(result=response.message.content)
