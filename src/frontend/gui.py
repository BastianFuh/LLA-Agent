import gradio as gr


def create_gui() -> gr.Blocks:
    with gr.Blocks() as demo:
        # Streaming functions need to be declared in the gr.Blocks() context
        from frontend.functions import chat

        chatbot = gr.Chatbot(
            type="messages",
            placeholder="<strong>Your Personal Yes-Man</strong><br>Ask Me Anything",
        )
        gr.ChatInterface(fn=chat, type="messages", chatbot=chatbot)

    return demo
