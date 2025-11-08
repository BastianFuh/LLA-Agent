import logging
from textwrap import TextWrapper

import gradio as gr
from gradio_toggle import Toggle

import frontend.tabs.shared as F
from backend.question_generator import tools as QGT
from frontend.tabs.messages.message_manager import MessageManager
from frontend.tabs.util import (
    create_audio_output,
    create_chatbot,
    create_textbox_with_audio_input,
)

logger = logging.getLogger(__name__)


def create_reading_comprehension_question_tab(
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
        with gr.Row():
            # Topic and Text
            with gr.Column():
                with gr.Group():
                    mode_switch = Toggle(
                        label=MessageManager()
                        .getMessages()
                        .label_listening_comprehension_switch(),
                        value=False,
                    )
                    show_text_button = gr.Button(
                        MessageManager().getMessages().button_show_text()
                    )
                with gr.Group():
                    topic = gr.Textbox(
                        "",
                        label=MessageManager().getMessages().label_topic(),
                        container=False,
                        placeholder=MessageManager()
                        .getMessages()
                        .placeholder_comprehension_topic(),
                        lines=1,
                        interactive=False,
                        show_label=False,
                    )

                    text = gr.TextArea(
                        show_label=False,
                        container=False,
                        placeholder=MessageManager()
                        .getMessages()
                        .placeholder_comprehension_text(),
                        interactive=False,
                        lines=11,
                        autoscroll=True,
                    )

                    question_create_button = gr.Button(
                        MessageManager().getMessages().button_generate_question(),
                        elem_classes="next-button",
                    )

            # Question and Answer
            with gr.Column():
                with gr.Group():
                    question = gr.Textbox(
                        label=MessageManager().getMessages().label_question(),
                        placeholder=MessageManager()
                        .getMessages()
                        .placeholder_comprehension_text(),
                        interactive=False,
                        show_copy_button=True,
                        lines=2,
                    )
                    answer = create_textbox_with_audio_input(
                        label=MessageManager().getMessages().label_answer(),
                        placeholder=MessageManager()
                        .getMessages()
                        .placeholder_answer_field(),
                        lines=3,
                        submit_btn=True,
                    )

                audio_player = create_audio_output(
                    tts_provider, language, topic, text, question
                )

        chatbot = create_chatbot(
            MessageManager().getMessages().placeholder_chatbot_evaluation()
        )
        chatbot_input = gr.Textbox(
            submit_btn=True,
            placeholder=MessageManager().getMessages().placeholder_chatbot_input(),
            show_label=False,
        )

        question_create_button.click(fn=F.clear, outputs=[chatbot]).then(
            create_reading_comprehension_question,
            [
                create_state,
                model,
                language,
                language_proficiency,
                difficulty,
                additional_information,
                mode_switch,
                tts_provider,
            ],
            [create_state, topic, text, question, answer, audio_player],
        )

        show_text_button.click(
            show_comprehension_text,
            [create_state, topic, text, question],
            [create_state, topic, text, question],
        )

        gr.on(
            triggers=[chatbot_input.submit],
            fn=F.reading_comprehension_chat,
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
            triggers=[answer.submit],
            fn=F.reading_comprehension_verifier,
            inputs=[
                topic,
                text,
                question,
                answer,
                chatbot,
                is_stream,
                audio_output,
                model,
                embedding_model,
                search_engine,
            ],
            outputs=[chatbot],
        )


async def create_reading_comprehension_question(
    state: dict,
    model: str,
    language: str,
    language_proficiency: str,
    difficulty: str,
    additional_information: str,
    mode_switch: bool,
    tts_provider: str,
):
    F.verify_input(language, language_proficiency, difficulty)

    state, question_generator = F.get_question_generator(state, model)

    yield (
        gr.skip(),
        gr.Textbox(value=""),
        gr.TextArea(value=""),
        gr.Textbox(value=""),
        gr.Textbox(value="", info=""),
        gr.skip(),
    )

    async for question_data in question_generator.generate_reading_comprehension(
        language,
        language_proficiency,
        difficulty,
        additional_information,
        mode_switch,
        tts_provider,
    ):
        topic = f"{question_data[QGT.READING_COMPREHENSION_TOPIC]}"
        text = f"{question_data[QGT.READING_COMPREHENSION_TEXT]}"
        question = f"{question_data[QGT.READING_COMPREHENSION_QUESTION]}"

        wrapped_text = TextWrapper(
            width=37, expand_tabs=False, replace_whitespace=False
        ).fill(text)

        if mode_switch:
            yield (
                state,
                gr.Textbox(value=topic, visible=False),
                gr.TextArea(value=wrapped_text, visible=False),
                gr.Textbox(value=question, visible=False),
                gr.Textbox(value="", info=""),
                question_data[QGT.AUDIO_DATA],
            )
        else:
            yield (
                state,
                gr.Textbox(value=topic),
                gr.TextArea(value=wrapped_text),
                gr.Textbox(value=question),
                gr.Textbox(value="", info=""),
                gr.skip(),
            )


async def show_comprehension_text(state, topic: str, text: str, question: str):
    yield (
        state,
        gr.Textbox(value=topic, visible=True),
        gr.TextArea(value=text, visible=True),
        gr.Textbox(value=question, visible=True),
    )
