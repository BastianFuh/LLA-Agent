import gradio as gr

import frontend.tabs.shared as F
from backend.question_generator import tools as QGT
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
                create_multiple_choice_questions,
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
                verify_multiple_choice_question,
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


async def create_multiple_choice_questions(
    state: gr.State,
    model: str,
    language: str,
    language_proficiency: str,
    difficulty: str,
    additional_information: str,
):
    F.verify_input(language, language_proficiency, difficulty)

    state, question_generator = F.get_question_generator(state, model)

    async for question_data in question_generator.generate_multiple_choice(
        language, language_proficiency, difficulty, additional_information
    ):
        state[QGT.QUESTION_ANSWER_INDEX] = question_data[QGT.QUESTION_ANSWER_INDEX]

        options = question_data[QGT.QUESTION_OPTIONS]

        question_text = f"{question_data[QGT.QUESTION_TEXT]}"

        yield (
            state,
            gr.Textbox(value=question_text, info=question_data[QGT.QUESTION_HINT]),
            gr.Checkbox(label=options[0], value=False, info=""),
            gr.Checkbox(label=options[1], value=False, info=""),
            gr.Checkbox(label=options[2], value=False, info=""),
            gr.Checkbox(label=options[3], value=False, info=""),
        )

    yield gr.skip()


async def verify_multiple_choice_question(state: dict, c1, c2, c3, c4):
    options = [c1, c2, c3, c4]

    if not state.__contains__(QGT.QUESTION_ANSWER_INDEX):
        gr.Info("Please generate a question first.")
        return gr.skip()

    correct_answer = state[QGT.QUESTION_ANSWER_INDEX]

    try:
        selected_answer = options.index(True)
    except ValueError:
        gr.Info("Please select a option")
        return gr.skip()

    updates = len(options) * [gr.Checkbox(info="")]

    if options[correct_answer]:
        updates[correct_answer] = gr.Checkbox(info="Correct Answer")
    else:
        updates[selected_answer] = gr.Checkbox(info="Wrong Answer")

    return updates
