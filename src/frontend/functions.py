from llama_index.core.workflow import Context
from workflow.events import LLMProgressEvent

from workflow.llm_workflow import ChatBotWorkfLow
from workflow.events import ChatBotStartEvent, AudioStreamEvent

from util import const

import gradio as gr


async def chat(message, history, is_stream: bool, audio_output: bool):
    workflow = ChatBotWorkfLow(timeout=60)

    ctx = Context(workflow)
    await ctx.set(const.IS_STREAM, is_stream)
    await ctx.set(const.AUDIO_OUTPUT, audio_output)
    response = ""

    start_event = ChatBotStartEvent(message=message, history=history)

    handler = workflow.run(start_event=start_event, ctx=ctx)

    async for event in handler.stream_events():
        if isinstance(event, LLMProgressEvent):
            response += event.response
            yield response

        if isinstance(event, AudioStreamEvent):
            # You can't return None here
            yield [
                response,
                gr.Audio(
                    event.audio_chunk,
                    type="numpy",
                    streaming=True,
                    autoplay=True,
                    interactive=False,
                ),
            ]
            # yield response, event.audio_chunk
