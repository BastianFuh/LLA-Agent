from llama_index.core.workflow import Context
from workflow.events import LLM_Progress_Event

from workflow.llm_workflow import LLM_FLow
from workflow.events import LLM_StartEvent


async def chat(message, history):
    workflow = LLM_FLow(timeout=60)
    is_stream = True

    ctx = Context(workflow)
    await ctx.set("is_stream", is_stream)
    response = ""

    start_event = LLM_StartEvent(message=message, history=history)

    if is_stream:
        handler = workflow.run(start_event=start_event, ctx=ctx)

        async for event in handler.stream_events():
            if isinstance(event, LLM_Progress_Event):
                response += event.response_delta
                yield response, None
    else:
        yield await workflow.run(start_event=start_event, ctx=ctx), None
