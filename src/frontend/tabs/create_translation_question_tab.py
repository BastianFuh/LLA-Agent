import gradio as gr

import frontend.functions as F
from frontend.tabs.util import (
    create_audio_output,
    create_chatbot,
    create_textbox_with_audio_input,
)


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
            F.create_translation_question,
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
            fn=F.translation_verifier_chat,
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
            fn=F.translation_verifier,
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
