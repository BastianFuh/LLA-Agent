import gradio as gr

import frontend.functions as F
from frontend.tabs.util import create_chatbot


def create_multiple_choice_questions_tab(
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
        state = gr.State({"selected_option": None})

        with gr.Row():
            with gr.Group():
                question_text = gr.TextArea(
                    "",
                    label="Question",
                    lines=1,
                    interactive=False,
                    placeholder="Your question will be generated here",
                )

                # Multiple checkboxes are used here instead of radio, or groupcheckbox because they should be in a
                # vertival row and this does not seem to be possible with the alternative options.

                question_options = [
                    gr.Checkbox(label="Answer 1"),
                    gr.Checkbox(label="Answer 2"),
                    gr.Checkbox(label="Answer 3"),
                    gr.Checkbox(label="Answer 4"),
                ]

                for checkbox in question_options:
                    checkbox.select(
                        F.process_select,
                        [state] + question_options,
                        [state] + question_options,
                    )

                    checkbox.change(
                        F.process_unselect, [state] + question_options, [state]
                    )

                with gr.Row():
                    question_create_button = gr.Button(
                        "Next", elem_classes="next-button", scale=1
                    )
                    question_submit_button = gr.Button(
                        "Submit", elem_classes="submit-custom-button", scale=2
                    )

            question_create_button.click(
                F.create_multiple_choice_questions,
                [
                    create_state,
                    model,
                    language,
                    language_proficiency,
                    difficulty,
                    additional_information,
                ],
                [create_state, question_text] + question_options,
            )

            question_submit_button.click(
                F.verify_multiple_choice_question,
                [create_state] + question_options,
                question_options,
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
