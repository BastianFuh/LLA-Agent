import gradio as gr

from llama_index.core.workflow import Context
from workflow.llm_workflow import LLM_FLow
from workflow.events import LLM_Progress_Event


def vote(data: gr.LikeData):
    if data.liked:
        print("You upvoted this response: " + data.value["value"])
    else:
        print("You downvoted this response: " + data.value["value"])


with gr.Blocks() as demo:
    workflow = LLM_FLow(timeout=60)

    async def test(message, history):
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

    chatbot = gr.Chatbot(
        type="messages",
        placeholder="<strong>Your Personal Yes-Man</strong><br>Ask Me Anything",
    )
    chatbot.like(vote, None, None)
    gr.ChatInterface(fn=test, type="messages", chatbot=chatbot)

if __name__ == "__main__":
    demo.launch()
