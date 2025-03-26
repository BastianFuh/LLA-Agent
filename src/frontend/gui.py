import gradio as gr

from frontend.functions import chat


def create_gui() -> gr.Blocks:
    with gr.Blocks(fill_height=True) as demo:
        # Streaming functions need to be declared in the gr.Blocks() context

        with gr.Tabs(selected=1):
            with gr.Tab("Options", id=0, scale=1):
                is_stream = gr.Checkbox(
                    label="Enable streaming for the chatbots output. (Currently outputs the entire thought process of the llm.)",
                    value=False,
                )
                audio_output = gr.Checkbox(
                    label="Enable audio output for the chatbot.", value=False
                )

            with gr.Tab("ChatBot", id=1, scale=1):
                chatbot = gr.Chatbot(
                    scale=2,
                    type="messages",
                    placeholder="<strong>Your Personal Language Learning Assistant</strong><br>Ask Me Anything",
                )
                gr.ChatInterface(
                    fn=chat,
                    type="messages",
                    chatbot=chatbot,
                    additional_inputs=[is_stream, audio_output],
                    fill_height=True,
                )

    return demo
