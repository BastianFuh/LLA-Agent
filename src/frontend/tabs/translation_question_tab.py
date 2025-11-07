import logging

import gradio as gr

import frontend.tabs.shared as F
import prompts
from backend.question_generator import tools as QGT
from frontend.tabs.util import (
    create_audio_output,
    create_chatbot,
    create_textbox_with_audio_input,
)

logger = logging.getLogger(__name__)


def create_translation_question_tab(
    browser_state: gr.BrowserState,
    create_state: gr.State,
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
):
    with gr.Column(
        scale=1,
    ):
        with gr.Group():
            question_text = gr.TextArea(
                "",
                label="Question",
                placeholder="Your question will be generated here",
                lines=1,
                interactive=False,
                show_copy_button=True,
            )

            answer_box = create_textbox_with_audio_input(
                show_label=False,
                submit_btn=True,
                placeholder="Type your answer...",
            )
            # question_submit_button = answer_box.submit_btn

            question_create_button = gr.Button(
                "Next", elem_classes="next-button", scale=1
            )

        create_audio_output(tts_provider, language, question_text)

        chatbot = create_chatbot(
            "<strong>The answers will be evaluted here</strong><br>You can also ask Me Anything"
        )

        chatbot_input = gr.Textbox(
            submit_btn=True, placeholder="Type a message...", show_label=False
        )

        question_create_button.click(fn=F.clear, outputs=[chatbot]).then(
            create_translation_question,
            [
                create_state,
                model,
                language,
                language_proficiency,
                difficulty,
                additional_information,
            ],
            [create_state, question_text, answer_box],
        )

        gr.on(
            triggers=[chatbot_input.submit],
            fn=translation_verifier_chat,
            inputs=[
                chatbot_input,
                chatbot,
                is_stream,
                audio_output,
                model,
                embedding_model,
                search_engine,
            ],
            outputs=[chatbot, chatbot_input],
        )

        gr.on(
            triggers=[answer_box.submit],
            fn=translation_verifier,
            inputs=[
                answer_box,
                chatbot,
                question_text,
                is_stream,
                audio_output,
                model,
                embedding_model,
                search_engine,
            ],
            outputs=[chatbot],
        )


def _create_translation_user_message(original: str, answer: str):
    return prompts.QUESTION_GENERATOR_TRANSLATION_REQUEST_PROMPT.format(
        original_text=original, translation=answer
    )


async def create_translation_question(
    state: gr.State,
    model: str,
    language: str,
    language_proficiency: str,
    difficulty: str,
    additional_information: str,
):
    F.verify_input(language, language_proficiency, difficulty)

    state, question_generator = F.get_question_generator(state, model)

    async for question_data in question_generator.generate_translation_question(
        language, language_proficiency, difficulty, additional_information
    ):
        question_text = f"{question_data[QGT.QUESTION_BASE_TEXT]}"

        yield (
            state,
            gr.Textbox(value=question_text),
            gr.Textbox(value="", info=""),
        )

    yield gr.skip()


async def translation_verifier(
    answer: str | None,
    history: list,
    original: str | None,
    is_stream: bool,
    audio_output: bool,
    model: str,
    embedding_model: str,
    search_engine: str,
):
    chat_prompts = {
        "function": prompts.QUESTION_GENERATOR_TRANSLATION_FUNCTION_CHATBOT_PROMPT,
        "react": prompts.QUESTION_GENERATOR_TRANSLATION_REACT_CHATBOT_PROMPT,
    }

    if answer is None:
        user_message = original
    else:
        user_message = _create_translation_user_message(original, answer)

    # Update shown history
    current_history = history.copy()
    current_history.append({"role": "user", "content": user_message})
    yield current_history

    async for event in F.chat(
        user_message,
        history,
        is_stream,
        audio_output,
        model,
        embedding_model,
        search_engine,
        chat_prompts,
    ):
        if isinstance(event, str):
            current_history.append({"role": "assistant", "content": event})

            yield current_history
        else:
            logger.warning(
                "Translation chat got an event which is currently not supported"
            )


async def translation_verifier_chat(
    message: str,
    history: list,
    is_stream: bool,
    audio_output: bool,
    model: str,
    embedding_model: str,
    search_engine: str,
):
    async for event in translation_verifier(
        None,
        history,
        message,
        is_stream,
        audio_output,
        model,
        embedding_model,
        search_engine,
    ):
        yield event, ""
