import logging

import gradio as gr
from gradio_toggle import Toggle

import frontend.tabs.shared as F
import prompts
from backend.question_generator import tools as QGT
from frontend.tabs.messages.message_manager import MessageManager
from frontend.tabs.util import create_chatbot, create_textbox_with_audio_input

logger = logging.getLogger(__name__)


def create_listening_comprehension_question_tab(
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
            with gr.Group():
                with gr.Column():
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
                    with gr.Row():
                        speaker_1 = gr.Textbox(
                            "",
                            label=MessageManager().getMessages().label_speaker_1(),
                            container=False,
                            placeholder=MessageManager()
                            .getMessages()
                            .placeholder_speaker_1_text(),
                            lines=1,
                            interactive=False,
                            show_label=False,
                        )

                        speaker_2 = gr.Textbox(
                            "",
                            label=MessageManager().getMessages().label_speaker_2(),
                            container=False,
                            placeholder=MessageManager()
                            .getMessages()
                            .placeholder_speaker_2_text(),
                            lines=1,
                            interactive=False,
                            show_label=False,
                        )

                    speaker_text = gr.Chatbot(
                        type="messages",
                        show_copy_button=True,
                        scale=1,
                        placeholder=MessageManager()
                        .getMessages()
                        .placeholder_comprehension_text(),
                    )

                    question_create_button = gr.Button(
                        MessageManager().getMessages().button_generate_question(),
                        elem_classes="next-button",
                    )

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

                    # Question and Answer
                with gr.Column():
                    with gr.Group():
                        question = gr.Textbox(
                            label=MessageManager().getMessages().label_question(),
                            placeholder=MessageManager()
                            .getMessages()
                            .question_placeholder(),
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

                    audio_player = gr.Audio(interactive=False, type="numpy")

        chatbot = create_chatbot(
            MessageManager().getMessages().placeholder_chatbot_evaluation()
        )
        chatbot_input = gr.Textbox(
            submit_btn=True,
            placeholder=MessageManager().getMessages().placeholder_chatbot_input(),
            show_label=False,
        )

        question_create_button.click(fn=F.clear, outputs=[chatbot]).then(
            create_listening_comprehension_question,
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
            [
                create_state,
                topic,
                speaker_1,
                speaker_2,
                speaker_text,
                question,
                answer,
                audio_player,
            ],
        )

        show_text_button.click(
            show_listening_comprehension_text,
            [create_state],
            [create_state, speaker_text],
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
            fn=listening_comprehension_verifier,
            inputs=[
                create_state,
                topic,
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


async def create_listening_comprehension_question(
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
        gr.Textbox(value=""),
        gr.Textbox(value=""),
        gr.Chatbot(value=[], type="messages"),
        gr.Textbox(value=""),
        gr.Textbox(value="", info=""),
        gr.skip(),
    )

    async for question_data in question_generator.generate_listening_comprehension(
        language,
        language_proficiency,
        difficulty,
        additional_information,
        mode_switch,
        tts_provider,
    ):
        state["listening_comprehension_data"] = question_data
        topic = f"{question_data[QGT.LISTENING_COMPREHENSION_TOPIC]}"
        speakers = question_data[QGT.LISTENING_COMPREHENSION_SPEAKERS]
        text = question_data[QGT.LISTENING_COMPREHENSION_TEXT]
        question = f"{question_data[QGT.LISTENING_COMPREHENSION_QUESTION]}"

        role_mapping = ["assistant", "user"]

        text_messages = [
            gr.ChatMessage(role=role_mapping[i % 2], content=text_segment["text"])
            for i, text_segment in enumerate(text)
        ]

        if mode_switch:
            yield (
                state,
                gr.Textbox(value=topic),
                gr.Textbox(value=speakers[0]),
                gr.Textbox(value=speakers[1]),
                gr.skip(),
                gr.Textbox(value=question),
                gr.Textbox(value="", info=""),
                question_data[QGT.AUDIO_DATA],
            )
        else:
            yield (
                state,
                gr.Textbox(value=topic),
                gr.Textbox(value=speakers[0]),
                gr.Textbox(value=speakers[1]),
                text_messages,
                gr.Textbox(value=question),
                gr.Textbox(value="", info=""),
                gr.skip(),
            )


async def show_listening_comprehension_text(state):
    role_mapping = ["assistant", "user"]

    data = state["listening_comprehension_data"]

    text = data[QGT.LISTENING_COMPREHENSION_TEXT]

    text_messages = [
        gr.ChatMessage(role=role_mapping[i % 2], content=text_segment["text"])
        for i, text_segment in enumerate(text)
    ]

    yield (
        state,
        text_messages,
    )


async def listening_comprehension_verifier(
    state: dict,
    topic: str,
    question: str,
    answer: str,
    history: list,
    is_stream: bool,
    audio_output: bool,
    model: str,
    embedding_model: str,
    search_engine: str,
):
    data = state["listening_comprehension_data"]

    text = data[QGT.LISTENING_COMPREHENSION_TEXT]

    complete_text = "\n".join(
        [f"{text_segment['speaker']}: {text_segment['text']}" for text_segment in text]
    )

    chat_prompts = {
        "function": prompts.READING_COMPREHENSION_FUNCTION_CHATBOT_PROMPT,
        "react": prompts.READING_COMPREHENSION_REACT_CHATBOT_PROMPT,
    }

    if answer is None:
        user_message = topic
    else:
        user_message = F.create_reading_comprehension_user_message(
            topic, complete_text, question, answer
        )

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
