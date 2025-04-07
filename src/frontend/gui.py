import gradio as gr
import frontend.functions as F

from util.const import OPTION_MODEL, OPTION_SEARCH_ENGINE, OPTION_EMBEDDING

import os


def create_gui() -> gr.Blocks:
    with gr.Blocks(fill_height=True) as demo:
        default_browser_state = {
            "language": "",
            "language_proficiency": "",
            "difficulty": "",
            "additional_information": "",
        }
        browser_state = gr.BrowserState(
            default_browser_state,
            storage_key="lla-agent",
            secret=os.getenv("LLA_AGENT_BROWSER_STATE_SECRET"),
        )

        with gr.Sidebar(position="right"):
            language = create_input(browser_state, False, "language", "Language")
            language_proficiency = create_input(
                browser_state, False, "language_proficiency", "Language Proficiency"
            )
            difficulty = create_input(browser_state, False, "difficulty", "Difficulty")
            additional_information = create_input(
                browser_state, True, "additional_information", "Additional Information"
            )

        with gr.Tabs(selected=1):
            with gr.Tab("Options", id=0, scale=1):
                model = gr.Dropdown(
                    choices=OPTION_MODEL,
                    value=OPTION_MODEL[5][1],
                    label="Chatbot Model",
                )

                embedding_model = gr.Dropdown(
                    choices=OPTION_EMBEDDING,
                    value=OPTION_EMBEDDING[0][1],
                    label="Embedding Model",
                )

                search_engine = gr.Dropdown(
                    choices=OPTION_SEARCH_ENGINE,
                    value=OPTION_SEARCH_ENGINE[0][1],
                    label="Search Engine",
                )

                is_stream = gr.Checkbox(
                    label="Enable streaming for the chatbots output. (Currently outputs the entire thought process of the llm.)",
                    value=False,
                )
                audio_output = gr.Checkbox(
                    label="Enable audio output for the chatbot.", value=False
                )

            with gr.Tab("ChatBot", id=1, scale=1):
                create_chatbot_tab(
                    browser_state,
                    is_stream,
                    audio_output,
                    model,
                    embedding_model,
                    search_engine,
                )

            with gr.Tab("Multiple Choice", id=2, scale=1):
                create_multiple_choice_questions(
                    browser_state,
                    is_stream,
                    audio_output,
                    model,
                    embedding_model,
                    search_engine,
                    language,
                    language_proficiency,
                    difficulty,
                    additional_information,
                )

            with gr.Tab("Simple Free Text", id=3, scale=1):
                create_free_text_questions(
                    browser_state,
                    is_stream,
                    audio_output,
                    model,
                    embedding_model,
                    search_engine,
                    language,
                    language_proficiency,
                    difficulty,
                    additional_information,
                )

    return demo


def create_chatbot_tab(
    browser_state: gr.BrowserState,
    is_stream,
    audio_output,
    model,
    embedding_model,
    search_engine,
):
    chatbot = gr.Chatbot(
        scale=2,
        type="messages",
        placeholder="<strong>Your Personal Language Learning Assistant</strong><br>Ask Me Anything",
    )
    gr.ChatInterface(
        fn=F.chat,
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


def create_input(
    browser_state: gr.BrowserState, textArea: bool, key_name: str, label: str
):
    def init(browser_state: gr.BrowserState):
        return browser_state[key_name]

    if textArea:
        field = gr.TextArea(
            value=init,
            label=label,
            inputs=[browser_state],
        )
    else:
        field = gr.Textbox(
            value=init,
            label=label,
            inputs=[browser_state],
        )

    def update(browser_state: gr.BrowserState, value: str):
        browser_state[key_name] = value
        return browser_state

    gr.on(
        triggers=[field.blur, field.submit],
        fn=update,
        inputs=[browser_state, field],
        outputs=[browser_state],
    )

    return field


def create_multiple_choice_questions(
    browser_state: gr.BrowserState,
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

        with gr.Column(scale=2, variant="panel"):
            question_text = gr.TextArea(
                "",
                label="Question",
                container=False,
                lines=1,
                interactive=False,
            )

            with gr.Column():
                question_options = [
                    gr.Checkbox(label=""),
                    gr.Checkbox(label=""),
                    gr.Checkbox(label=""),
                    gr.Checkbox(label=""),
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
                question_create_button = gr.Button("Next")
                question_submit_button = gr.Button("Submit")

            question_create_button.click(
                F.create_multiple_choice_questions,
                [
                    state,
                    model,
                    language,
                    language_proficiency,
                    difficulty,
                    additional_information,
                ],
                [question_text] + question_options,
            )

            question_submit_button.click(
                F.verify_multiple_choice_question,
                [state] + question_options,
                question_options,
            )

        chatbot = gr.Chatbot(
            scale=1,
            type="messages",
            placeholder="<strong>The answers will be evaluted here</strong><br>You can also ask Me Anything",
        )
        gr.ChatInterface(
            fn=F.chat,
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


def create_free_text_questions(
    browser_state: gr.BrowserState,
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

        with gr.Column(scale=2, variant="panel"):
            question_text = gr.TextArea(
                "",
                label="Question",
                container=False,
                lines=1,
                interactive=False,
            )

            with gr.Column():
                answer_box = gr.Textbox(label="Answer")

            with gr.Row():
                question_create_button = gr.Button("Next")
                question_submit_button = gr.Button("Submit")
                question_show_answer = gr.Button("Show Answer")

            question_create_button.click(
                F.create_free_text_question,
                [
                    state,
                    model,
                    language,
                    language_proficiency,
                    difficulty,
                    additional_information,
                ],
                [question_text, answer_box],
            )

            gr.on(
                triggers=[question_submit_button.click, answer_box.submit],
                fn=F.verify_free_text_question,
                inputs=[state, answer_box],
                outputs=[answer_box],
            )

            question_show_answer.click(F.show_free_text_answer, [state], [answer_box])

        chatbot = gr.Chatbot(
            scale=1,
            type="messages",
            placeholder="<strong>The answers will be evaluted here</strong><br>You can also ask Me Anything",
        )
        gr.ChatInterface(
            fn=F.chat,
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
