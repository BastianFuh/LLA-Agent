import gradio as gr

import frontend.tabs.shared as F
from backend.question_generator import tools as QGT
from frontend.tabs.messages.message_manager import MessageManager
from frontend.tabs.util import create_chatbot


def create_free_text_questions_tab(
    browser_state: gr.BrowserState,
    create_state: gr.State,
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
    with gr.Column(
        scale=1,
    ):
        with gr.Group():
            ###
            # Layout
            ###
            question_text = gr.TextArea(
                label=MessageManager().getMessages().label_question(),
                placeholder=MessageManager().getMessages().question_placeholder(),
                lines=1,
                interactive=False,
            )

            question_create_button = gr.Button(
                MessageManager().getMessages().button_generate_question(),
                elem_classes="next-button",
                scale=1,
            )

            answer_box = gr.Textbox(
                show_label=False,
                placeholder=MessageManager().getMessages().placeholder_answer_field(),
                submit_btn=True,
            )

            # question_show_answer = gr.Button(
            #    MessageManager().getMessages().button_validate_answer(),
            #    elem_classes="show-answer-button",
            #    scale=1,
            # )

            ###
            # Functionality
            ###

            question_create_button.click(
                create_free_text_question,
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
                triggers=[answer_box.submit],
                fn=verify_free_text_question,
                inputs=[create_state, answer_box],
                outputs=[answer_box],
            )

            # question_show_answer.click(
            #     show_free_text_answer, [create_state], [answer_box]
            # )

        chatbot = create_chatbot()
        gr.ChatInterface(
            fn=F.assistant_chat,
            type="messages",
            chatbot=chatbot,
            additional_inputs=[
                is_stream,
                audio_output,
                model,
                embedding_model,
                search_engine,
            ],
            fill_height=True,
        )


async def show_free_text_answer(state: dict):
    if not state.__contains__(QGT.QUESTION_ANSWER):
        gr.Info(MessageManager().getMessages().info_generate_question_first())
        return gr.skip()

    return gr.Textbox(info=state[QGT.QUESTION_ANSWER])


async def verify_free_text_question(state: dict, answer: str):
    if not state.__contains__(QGT.QUESTION_ANSWER):
        gr.Info(MessageManager().getMessages().info_generate_question_first())
        return gr.skip()

    correct_answer = state[QGT.QUESTION_ANSWER]

    if correct_answer == answer:
        update_answer_field = gr.Textbox(
            info=MessageManager().getMessages().info_correct_answer()
        )
    else:
        update_answer_field = gr.Textbox(
            info=MessageManager().getMessages().info_incorrect_answer()
        )

    return update_answer_field


async def create_free_text_question(
    state: gr.State,
    model: str,
    language: str,
    language_proficiency: str,
    difficulty: str,
    additional_information: str,
):
    F.verify_input(language, language_proficiency, difficulty)

    state, question_generator = F.get_question_generator(state, model)

    async for question_data in question_generator.generate_free_text(
        language, language_proficiency, difficulty, additional_information
    ):
        state[QGT.QUESTION_ANSWER] = question_data[QGT.QUESTION_ANSWER]

        question_text = f"{question_data[QGT.QUESTION_TEXT]}"

        yield (
            state,
            gr.Textbox(value=question_text, info=question_data[QGT.QUESTION_HINT]),
            gr.Textbox(value="", info=""),
        )

    yield gr.skip()
