import gradio as gr

from llama_index.core.workflow import Context
from workflow.llm_workflow import LLM_FLow


async def test(message, history):
    print(history)

    ctx = Context(workflow)
    await ctx.set("is_stream", False)

    return await workflow.run(message=message, history=history, ctx=ctx)


def vote(data: gr.LikeData):
    if data.liked:
        print("You upvoted this response: " + data.value["value"])
    else:
        print("You downvoted this response: " + data.value["value"])


with gr.Blocks() as demo:
    workflow = LLM_FLow()

    chatbot = gr.Chatbot(
        type="messages",
        placeholder="<strong>Your Personal Yes-Man</strong><br>Ask Me Anything",
    )
    chatbot.like(vote, None, None)
    gr.ChatInterface(fn=test, type="messages", chatbot=chatbot)

if __name__ == "__main__":
    demo.launch()
