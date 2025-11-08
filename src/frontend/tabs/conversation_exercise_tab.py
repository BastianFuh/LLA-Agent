import gradio as gr

import frontend.tabs.shared as F
import prompts
from frontend.tabs.messages.message_manager import MessageManager


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
        placeholder=MessageManager().getMessages().placeholder_chatbot(),
    )

    gr.ChatInterface(
        fn=conversation_chat,
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


async def conversation_chat(
    message: str,
    history: list,
    is_stream: bool,
    audio_output: bool,
    model: str,
    embedding_model: str,
    search_engine: str,
    language: str,
    language_proficiency: str,
    difficulty: str,
    additional_information: str,
):
    F.verify_input(language, language_proficiency, difficulty)

    prompt = prompts.CONVERSATION_BOT_FUNCTION_PROMPT

    options = {
        "language": language,
        "language_proficiency": language_proficiency,
        "difficulty": difficulty,
        "additional_information": additional_information,
    }

    prompt = prompt.format(**options)

    async for event in F.chat(
        message,
        history,
        is_stream,
        audio_output,
        model,
        embedding_model,
        search_engine,
        prompt,
    ):
        yield event
