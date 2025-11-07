import gradio as gr
from gradio_toggle import Toggle

import frontend.functions as F
from frontend.tabs.util import create_chatbot, create_textbox_with_audio_input


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
                        label="Topic",
                        container=False,
                        placeholder="The topic of the text",
                        lines=1,
                        interactive=False,
                        show_label=False,
                    )
                    with gr.Row():
                        speaker_1 = gr.Textbox(
                            "",
                            label="Speaker 1",
                            container=False,
                            placeholder="The name of the first peaker",
                            lines=1,
                            interactive=False,
                            show_label=False,
                        )

                        speaker_2 = gr.Textbox(
                            "",
                            label="Speaker 2",
                            container=False,
                            placeholder="The name of the second speaker",
                            lines=1,
                            interactive=False,
                            show_label=False,
                        )

                    speaker_text = gr.Chatbot(
                        type="messages",
                        show_copy_button=True,
                        scale=1,
                        placeholder="The text of the listening comprehension",
                    )

                    question_create_button = gr.Button(
                        "Next", elem_classes="next-button"
                    )

            with gr.Column():
                with gr.Group():
                    mode_switch = Toggle(label="Listening Comprehension", value=False)
                    show_text_button = gr.Button("Show Text")

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

                    audio_player = gr.Audio(interactive=False, type="numpy")

        chatbot = create_chatbot(
            "<strong>The answers will be evaluted here</strong><br>You can also ask Me Anything"
        )
        chatbot_input = gr.Textbox(
            submit_btn=True, placeholder="Type a message...", show_label=False
        )

        question_create_button.click(fn=F.clear, outputs=[chatbot]).then(
            F.create_listening_comprehension_question,
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
            F.show_listening_comprehension_text,
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
            fn=F.listening_comprehension_verifier,
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
