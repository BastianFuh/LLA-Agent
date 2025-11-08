import os

import gradio as gr

from frontend.tabs import (
    create_conversation_exercise_tab,
    create_free_text_questions_tab,
    create_listening_comprehension_question_tab,
    create_multiple_choice_questions_tab,
    create_reading_comprehension_question_tab,
    create_translation_question_tab,
)
from frontend.tabs.messages.message_manager import MessageManager
from frontend.tabs.util import (
    create_checkbox_input,
    create_dropdown_input,
    create_text_input,
)
from util.const import (
    OPTION_SEARCH_ENGINE,
    TTS_CHATTERBOX,
    TTS_ELEVENLABS,
    TTS_FISH_AUDIO,
    TTS_KOKORO,
    TTS_OPENAI,
)
from util.model import get_embedding_models, get_llm_models


def create_gui() -> gr.Blocks:
    default_browser_state = {
        "language": "",
        "language_proficiency": "",
        "difficulty": "",
        "additional_information": "",
        "option_model": list(get_llm_models().keys())[0],
        "option_embedding_model": list(get_embedding_models().keys())[0],
        "option_search_engine": OPTION_SEARCH_ENGINE[0][1],
        "option_is_stream": False,
        "option_audio_output": False,
    }
    browser_state = gr.BrowserState(
        default_browser_state,
        storage_key="lla-agent",
        secret=os.getenv("LLA_AGENT_BROWSER_STATE_SECRET"),
    )

    create_state = gr.State({})

    with gr.Sidebar(
        label=MessageManager().getMessages().label_settings(), position="right"
    ):
        with gr.Group():
            language = create_text_input(
                browser_state,
                False,
                "language",
                MessageManager().getMessages().label_language(),
            )
            language_proficiency = create_text_input(
                browser_state,
                False,
                "language_proficiency",
                MessageManager().getMessages().label_language_proficiency(),
            )
            difficulty = create_text_input(
                browser_state,
                False,
                "difficulty",
                MessageManager().getMessages().label_difficulty(),
            )
            additional_information = create_text_input(
                browser_state,
                True,
                "additional_information",
                MessageManager().getMessages().label_additional_information(),
            )

    with gr.Tabs(selected=1):
        with gr.Tab(MessageManager().getMessages().tab_label_options(), id=0, scale=1):
            options = list(get_llm_models().keys())
            model = create_dropdown_input(
                browser_state,
                options,
                "option_model",
                MessageManager().getMessages().label_model(),
            )

            embedding_model = create_dropdown_input(
                browser_state,
                list(get_embedding_models().keys()),
                "option_embedding_model",
                MessageManager().getMessages().label_embedding_model(),
            )

            search_engine = create_dropdown_input(
                browser_state,
                OPTION_SEARCH_ENGINE,
                "option_search_engine",
                MessageManager().getMessages().label_search_engine(),
            )

            tts_provider = create_dropdown_input(
                browser_state,
                [
                    TTS_KOKORO,
                    TTS_ELEVENLABS,
                    TTS_OPENAI,
                    TTS_FISH_AUDIO,
                    TTS_CHATTERBOX,
                ],
                "tts_provider",
                MessageManager().getMessages().label_tts_provider(),
            )

            is_stream = create_checkbox_input(
                browser_state,
                "option_is_stream",
                "Enable streaming for the chatbots output. (Currently outputs the entire thought process of the llm.)",
            )

            audio_output = create_checkbox_input(
                browser_state,
                "option_audio_output",
                "Enable audio output for the chatbot.",
            )

        with gr.Tab(
            MessageManager().getMessages().tab_label_conversation_exercise(),
            id=1,
            scale=1,
        ):
            create_conversation_exercise_tab(
                browser_state,
                is_stream,
                audio_output,
                model,
                embedding_model,
                search_engine,
                language,
                language_proficiency,
                difficulty,
                additional_information,
            )

        with gr.Tab(
            MessageManager().getMessages().tab_label_multiple_choice_question(),
            id=2,
            scale=1,
        ):
            create_multiple_choice_questions_tab(
                browser_state,
                create_state,
                is_stream,
                audio_output,
                model,
                embedding_model,
                search_engine,
                language,
                language_proficiency,
                difficulty,
                additional_information,
            )

        with gr.Tab(
            MessageManager().getMessages().tab_label_fill_in_blank_question(),
            id=3,
            scale=1,
        ):
            create_free_text_questions_tab(
                browser_state,
                create_state,
                is_stream,
                audio_output,
                model,
                embedding_model,
                search_engine,
                language,
                language_proficiency,
                difficulty,
                additional_information,
            )

        with gr.Tab(
            MessageManager().getMessages().tab_label_translation_question(),
            id=4,
            scale=1,
        ):
            create_translation_question_tab(
                browser_state,
                create_state,
                is_stream,
                audio_output,
                model,
                embedding_model,
                search_engine,
                tts_provider,
                language,
                language_proficiency,
                difficulty,
                additional_information,
            )

        with gr.Tab(
            MessageManager().getMessages().tab_label_comprehension_monologue_question(),
            id=5,
            scale=1,
        ):
            create_reading_comprehension_question_tab(
                browser_state,
                create_state,
                is_stream,
                audio_output,
                model,
                embedding_model,
                search_engine,
                tts_provider,
                language,
                language_proficiency,
                difficulty,
                additional_information,
            )

        with gr.Tab(
            MessageManager().getMessages().tab_label_comprehension_dialogue_question(),
            id=6,
            scale=1,
        ):
            create_listening_comprehension_question_tab(
                browser_state,
                create_state,
                is_stream,
                audio_output,
                model,
                embedding_model,
                search_engine,
                tts_provider,
                language,
                language_proficiency,
                difficulty,
                additional_information,
            )
