import numpy as np

from llama_index.core.workflow import (
    StopEvent,
    Workflow,
    Context,
    step,
)
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.base.llms.types import ChatMessage

from llama_index.core.agent.react import ReActChatFormatter
from llama_index.core.agent.workflow import ReActAgent

from llama_index.core.agent.workflow.workflow_events import AgentStream

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
from workflow import utils as w_utils

import logging

from pathlib import Path


def build_message(message: str, history: list[dict]):
    llm_message = list()

    for element in history:
        llm_message.append(
            ChatMessage(role=element["role"], content=element["content"])
        )

    if message is not None:
        llm_message.append(ChatMessage(role="user", content=message))

    return llm_message if len(llm_message) > 0 else None


class ChatBotWorkfLow(Workflow):
    PROMPT_FILE = (
        (Path(__file__).parents[0] / Path("prompt.md"))
        .open("r", encoding="utf-8")
        .read()
    )

    def __init__(self, prompt: str = PROMPT_FILE):
        self.prompt = prompt

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

        model = await ctx.get(const.MODEL)

        model_information = const.OPTION_MODEL[model]

        model_name = model_information[0]
        provider = model_information[1]

        logging.info(f"Using model: {model_name} from provider {provider} ")
        llm = w_utils.get_llm(model)

        llm_tools = await w_utils.get_llms_tools(ctx)

        agent = ReActAgent(
            name="Chatbot Agent",
            description="Todo",
            tools=llm_tools,
            llm=llm,
            formatter=ReActChatFormatter.from_defaults(system_header=self.prompt),
        )

        agent_ctx = Context(agent)

        is_stream = await ctx.get(const.IS_STREAM, default=False)

        # TODO: Check if there is a way to stream the ouput of an agent without the thought process
        if is_stream:
            handler = agent.run(
                ev.message, ctx=agent_ctx, chat_history=build_message(None, ev.history)
            )

            async for handler_ev in handler.stream_events():
                if isinstance(handler_ev, AgentStream):
                    ctx.write_event_to_stream(
                        LLMProgressEvent(response=handler_ev.delta)
                    )

            response = await handler
        else:
            response = await agent.run(
                ev.message, ctx=agent_ctx, chat_history=build_message(None, ev.history)
            )

        ctx.write_event_to_stream(LLMProgressEvent(response=response.response.content))

        return LLMFinishedEvent(result=response.response.content)

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
