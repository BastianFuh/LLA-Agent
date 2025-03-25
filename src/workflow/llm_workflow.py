import numpy as np

from llama_index.core.workflow import (
    StopEvent,
    Workflow,
    Context,
    step,
)
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.base.llms.types import ChatMessage

from openai import AsyncOpenAI

from workflow.events import (
    ChatBotStartEvent,
    LLMStartEvent,
    LLMProgressEvent,
    LLMFinishedEvent,
    AudioFinishedEvent,
    AudioStartEvent,
    AudioStreamEvent,
)

from util import const

import logging


def build_message(message: str, history: list[dict]):
    llm_message = list()
    for element in history:
        llm_message.append(
            ChatMessage(role=element["role"], content=element["content"])
        )

    llm_message.append(ChatMessage(role="user", content=message))

    return llm_message


class ChatBotWorkfLow(Workflow):
    llm = OpenRouter(model="deepseek/deepseek-chat-v3-0324:free")

    @step
    async def control(
        self,
        ctx: Context,
        ev: ChatBotStartEvent | LLMFinishedEvent | AudioFinishedEvent,
    ) -> StopEvent | LLMStartEvent:
        if isinstance(ev, ChatBotStartEvent):
            if ev.message is None or ev.message == "":
                logging.error(
                    "ChatBot Workflow could not be started because of missing data"
                )
                return StopEvent()

            logging.info("Started ChatBot Workflow")

            return LLMStartEvent(message=ev.message, history=ev.history)

        events = ctx.collect_events(ev, [LLMFinishedEvent, AudioFinishedEvent])

        if events is None:
            return None

        logging.info("Finished ChatBot Workflow")

        return StopEvent()

    @step
    async def llm_step(self, ctx: Context, ev: LLMStartEvent) -> LLMFinishedEvent:
        logging.info("Started llm request")

        is_stream = await ctx.get(const.IS_STREAM, default=False)

        if is_stream:
            gen = await self.llm.astream_chat(build_message(ev.message, ev.history))

            async for response in gen:
                ctx.write_event_to_stream(LLMProgressEvent(response=response.delta))
        else:
            response = await self.llm.achat(build_message(ev.message, ev.history))
            ctx.write_event_to_stream(
                LLMProgressEvent(response=response.message.content)
            )

        return LLMFinishedEvent(result=response.message.content)

    @step
    async def audio_prepare(
        self, ctx: Context, ev: LLMFinishedEvent
    ) -> AudioStartEvent | AudioFinishedEvent:
        audio_output = await ctx.get(const.AUDIO_OUTPUT, default=False)

        if not audio_output:
            return AudioFinishedEvent()

        logging.info("Preparing audio generation")

        return AudioStartEvent(text=ev.result)

    @step
    async def audio_process(
        self, ctx: Context, ev: AudioStartEvent
    ) -> AudioFinishedEvent:
        logging.info("Generating audio")

        client = AsyncOpenAI()
        response = await client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="onyx",
            input=ev.text,
            instructions="Your pronounciation should be really precise.",
            response_format="pcm",
        )

        ctx.write_event_to_stream(
            AudioStreamEvent(
                audio_chunk=(24000, np.frombuffer(response.content, dtype=np.int16))
            )
        )

        return AudioFinishedEvent()
