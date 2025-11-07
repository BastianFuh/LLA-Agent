import gradio as gr

import frontend.functions as F
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
            question_text = gr.TextArea(
                label="Question",
                placeholder="Your question will be generated here",
                lines=1,
                interactive=False,
            )

            answer_box = gr.Textbox(
                show_label=False,
                placeholder="Type your answer...",
                submit_btn=True,
            )

            with gr.Row():
                question_create_button = gr.Button(
                    "Next", elem_classes="next-button", scale=1
                )
                question_show_answer = gr.Button(
                    "Show Answer", elem_classes="show-answer-button", scale=1
                )

            question_create_button.click(
                F.create_free_text_question,
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
                fn=F.verify_free_text_question,
                inputs=[create_state, answer_box],
                outputs=[answer_box],
            )

            question_show_answer.click(
                F.show_free_text_answer, [create_state], [answer_box]
            )

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
