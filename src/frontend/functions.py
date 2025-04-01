from llama_index.core.workflow import Context

from workflow.events import LLMProgressEvent

from workflow.chatbot.chatbot_workflow import ChatBotWorkfLow
from workflow.events import ChatBotStartEvent, AudioStreamEvent, LLMFinishedEvent

from util import const

import gradio as gr


async def chat(
    message: str,
    history: dict,
    is_stream: bool,
    audio_output: bool,
    model: str,
    embedding_model: str,
    search_engine: str,
):
    workflow = ChatBotWorkfLow(timeout=None)

    ctx = Context(workflow)
    await ctx.set(const.IS_STREAM, is_stream)
    await ctx.set(const.AUDIO_OUTPUT, audio_output)
    await ctx.set(const.MODEL, model)
    await ctx.set(const.EMBEDDING_MODEL, embedding_model)
    await ctx.set(const.SEARCH_ENGINE, search_engine)
    response = ""

    start_event = ChatBotStartEvent(message=message, history=history)

    handler = workflow.run(start_event=start_event, ctx=ctx)

    async for event in handler.stream_events():
        if isinstance(event, LLMProgressEvent):
            response += event.response
            yield response

        if isinstance(event, AudioStreamEvent):
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


def process_select(state, c1, c2, c3, c4):
    options = [c1, c2, c3, c4]

    assert options.count(True) <= 2, "The option select broke"

    if options.count(True) > 1:
        options[state["selected_option"]] = False

    for i, o in enumerate(options):
        if o:
            state["selected_option"] = i

    return state, options[0], options[1], options[2], options[3]


def process_unselect(state, c1, c2, c3, c4):
    options = [c1, c2, c3, c4]

    if options.count(False) == len(options):
        state["selected_option"] = None

    return state
