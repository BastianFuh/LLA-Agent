from llama_index.core.workflow import Context
from workflow.events import LLM_Progress_Event

from workflow.llm_workflow import LLM_FLow


async def chat(message, history):
    workflow = LLM_FLow(timeout=60)
    is_stream = True

    ctx = Context(workflow)
    await ctx.set("is_stream", is_stream)
    response = ""
    if is_stream:
        handler = workflow.run(message=message, history=history, ctx=ctx)

        async for event in handler.stream_events():
            if isinstance(event, LLM_Progress_Event):
                response += event.response_delta
                yield response
    else:
        yield await workflow.run(message=message, history=history, ctx=ctx)
