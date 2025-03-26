import gradio as gr


def create_gui() -> gr.Blocks:
    with gr.Blocks() as demo:
        # Streaming functions need to be declared in the gr.Blocks() context
        from frontend.functions import chat

        with gr.Tab("Options"):
            is_stream = gr.Checkbox(
                label="Enable streaming for the chatbots output. (Currently outputs the entire thought process of the llm.)",
                value=False,
            )
            audio_output = gr.Checkbox(
                label="Enable audio output for the chatbot.", value=False
            )

        with gr.Tab("ChatBot"):
            chatbot = gr.Chatbot(
                type="messages",
                placeholder="<strong>Your Personal Yes-Man</strong><br>Ask Me Anything",
                scale=0,
            )
            gr.ChatInterface(
                fn=chat,
                type="messages",
                chatbot=chatbot,
                additional_inputs=[is_stream, audio_output],
            )

    return demo
