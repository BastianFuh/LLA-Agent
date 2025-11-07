import gradio as gr

import frontend.functions as F


def create_conversation_exercise_tab(
    browser_state: gr.BrowserState,
    is_stream,
    audio_output,
    model,
    embedding_model,
    search_engine,
    language,
    language_proficiency,
    difficulty,
    additional_information,
):
    chatbot = gr.Chatbot(
        type="messages",
        scale=1,
        placeholder="<strong>Your Personal Language Learning Assistant</strong><br>Ask Me Anything",
    )
    gr.ChatInterface(
        fn=F.conversation_chat,
        type="messages",
        chatbot=chatbot,
        additional_inputs=[
            is_stream,
            audio_output,
            model,
            embedding_model,
            search_engine,
            language,
            language_proficiency,
            difficulty,
            additional_information,
        ],
        fill_height=True,
    )
