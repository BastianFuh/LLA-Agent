import gradio as gr
from gradio_toggle import Toggle

import frontend.functions as F
from frontend.tabs.util import (
    create_audio_output,
    create_chatbot,
    create_textbox_with_audio_input,
)


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
                    mode_switch = Toggle(label="Listening Comprehension", value=False)
                    show_text_button = gr.Button("Show Text")
                with gr.Group():
                    topic = gr.Textbox(
                        "",
                        label="Topic",
                        container=False,
                        placeholder="The topic of the text",
                        lines=1,
                        interactive=False,
                        show_label=False,
                    )

                    text = gr.TextArea(
                        show_label=False,
                        container=False,
                        placeholder="The text of the reading comprehension",
                        interactive=False,
                        lines=11,
                        autoscroll=True,
                    )

                    question_create_button = gr.Button(
                        "Next", elem_classes="next-button"
                    )

            # Question and Answer
            with gr.Column():
                with gr.Group():
                    question = gr.Textbox(
                        label="Question",
                        placeholder="Your question will be generated here",
                        interactive=False,
                        show_copy_button=True,
                        lines=2,
                    )
                    answer = create_textbox_with_audio_input(
                        label="Answer",
                        placeholder="Type your answer...",
                        lines=3,
                        submit_btn=True,
                    )

                audio_player = create_audio_output(
                    tts_provider, language, topic, text, question
                )

        chatbot = create_chatbot(
            "<strong>The answers will be evaluted here</strong><br>You can also ask Me Anything"
        )
        chatbot_input = gr.Textbox(
            submit_btn=True, placeholder="Type a message...", show_label=False
        )

        question_create_button.click(fn=F.clear, outputs=[chatbot]).then(
            F.create_reading_comprehension_question,
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
            F.show_comprehension_text,
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
