import gradio as gr
import frontend.functions as F

from util.const import OPTION_MODEL, OPTION_SEARCH_ENGINE, OPTION_EMBEDDING

import os


def create_gui() -> gr.Blocks:
    default_browser_state = {
        "language": "",
        "language_proficiency": "",
        "difficulty": "",
        "additional_information": "",
        "option_model": list(OPTION_MODEL.keys())[0],
        "option_embedding_model": OPTION_EMBEDDING[0][1],
        "option_search_engine": OPTION_SEARCH_ENGINE[0][1],
        "option_is_stream": False,
        "option_audio_output": False,
    }
    browser_state = gr.BrowserState(
        default_browser_state,
        storage_key="lla-agent",
        secret=os.getenv("LLA_AGENT_BROWSER_STATE_SECRET"),
    )

    create_state = gr.State({})

    with gr.Sidebar(position="right"):
        language = create_text_input(browser_state, False, "language", "Language")
        language_proficiency = create_text_input(
            browser_state, False, "language_proficiency", "Language Proficiency"
        )
        difficulty = create_text_input(browser_state, False, "difficulty", "Difficulty")
        additional_information = create_text_input(
            browser_state, True, "additional_information", "Additional Information"
        )

    with gr.Tabs(selected=1):
        with gr.Tab("Options", id=0, scale=1):
            options = list(OPTION_MODEL.keys())
            model = create_dropdown_input(
                browser_state, options, "option_model", "Chatbot Model"
            )

            embedding_model = create_dropdown_input(
                browser_state,
                OPTION_EMBEDDING,
                "option_embedding_model",
                "Embedding Model",
            )

            search_engine = create_dropdown_input(
                browser_state,
                OPTION_SEARCH_ENGINE,
                "option_search_engine",
                "Search Engine",
            )

            is_stream = create_checkbox_input(
                browser_state,
                "option_is_stream",
                "Enable streaming for the chatbots output. (Currently outputs the entire thought process of the llm.)",
            )

            audio_output = create_checkbox_input(
                browser_state,
                "option_audio_output",
                "Enable audio output for the chatbot.",
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
                create_state,
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
                create_state,
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

        with gr.Tab("Translation", id=4, scale=1):
            create_translation_question(
                browser_state,
                create_state,
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

        with gr.Tab("Reading Comprehension", id=5, scale=1):
            create_reading_comprehension_question(
                browser_state,
                create_state,
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
        fn=F.basic_chat,
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


def create_text_input(
    browser_state: gr.BrowserState, textArea: bool, key_name: str, label: str
):
    def init(browser_state: gr.BrowserState):
        return browser_state[key_name]

    def update(browser_state: gr.BrowserState, value: str):
        browser_state[key_name] = value
        return browser_state

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

    gr.on(
        triggers=[field.blur, field.submit],
        fn=update,
        inputs=[browser_state, field],
        outputs=[browser_state],
    )

    return field


def create_dropdown_input(
    browser_state: gr.BrowserState, choices: list, key_name: str, label: str
):
    def init(browser_state: gr.BrowserState):
        return browser_state[key_name]

    def update(browser_state: gr.BrowserState, value: str):
        browser_state[key_name] = value
        return browser_state

    field = gr.Dropdown(
        value=init, label=label, choices=choices, inputs=[browser_state]
    )

    gr.on(
        triggers=[field.blur, field.select],
        fn=update,
        inputs=[browser_state, field],
        outputs=[browser_state],
    )

    return field


def create_checkbox_input(browser_state: gr.BrowserState, key_name: str, label: str):
    def init(browser_state: gr.BrowserState):
        return browser_state[key_name]

    def update(browser_state: gr.BrowserState, value: bool):
        browser_state[key_name] = value
        return browser_state

    field = gr.Checkbox(value=init, inputs=[browser_state], label=label)

    gr.on(
        triggers=[field.change],
        fn=update,
        inputs=[browser_state, field],
        outputs=[browser_state],
    )

    return field


def create_multiple_choice_questions(
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
            with gr.Column(variant="panel"):
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

        chatbot = gr.Chatbot(
            type="messages",
            placeholder="<strong>The answers will be evaluted here</strong><br>You can also ask Me Anything",
        )
        gr.ChatInterface(
            fn=F.basic_chat,
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
        with gr.Row():
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
                    triggers=[question_submit_button.click, answer_box.submit],
                    fn=F.verify_free_text_question,
                    inputs=[create_state, answer_box],
                    outputs=[answer_box],
                )

                question_show_answer.click(
                    F.show_free_text_answer, [create_state], [answer_box]
                )

        chatbot = gr.Chatbot(
            type="messages",
            placeholder="<strong>The answers will be evaluted here</strong><br>You can also ask Me Anything",
        )
        gr.ChatInterface(
            fn=F.basic_chat,
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


def create_translation_question(
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
        with gr.Row():
            with gr.Column(variant="panel"):
                question_text = gr.TextArea(
                    "",
                    label="Question",
                    container=False,
                    lines=1,
                    interactive=False,
                    show_copy_button=True,
                )

                with gr.Column():
                    answer_box = gr.Textbox(label="Answer")

                with gr.Row():
                    question_create_button = gr.Button("Next")
                    question_submit_button = gr.Button("Submit")

                question_create_button.click(
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

        chatbot = gr.Chatbot(
            type="messages",
            show_copy_button=True,
            resizable=True,
            min_height=700,
            placeholder="<strong>The answers will be evaluted here</strong><br>You can also ask Me Anything",
        )
        chatbot_input = gr.Textbox(
            submit_btn=True, placeholder="Type a message...", show_label=False
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
            triggers=[question_submit_button.click, answer_box.submit],
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


def create_reading_comprehension_question(
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
        with gr.Row():
            with gr.Column():
                topic = gr.Textbox(
                    "",
                    label="topic",
                    container=False,
                    lines=1,
                    interactive=False,
                    show_copy_button=True,
                )

                with gr.Column():
                    text = gr.TextArea(
                        label="Text",
                        interactive=False,
                        show_copy_button=True,
                    )
                    question = gr.Textbox(
                        label="Question",
                        interactive=False,
                        show_copy_button=True,
                    )
                    answer = gr.TextArea(label="Answer", lines=3)

                with gr.Row():
                    question_create_button = gr.Button("Next")
                    question_submit_button = gr.Button("Submit")

                question_create_button.click(
                    F.create_reading_comprehension_question,
                    [
                        create_state,
                        model,
                        language,
                        language_proficiency,
                        difficulty,
                        additional_information,
                    ],
                    [create_state, topic, text, question, answer],
                )

        chatbot = gr.Chatbot(
            type="messages",
            show_copy_button=True,
            resizable=True,
            min_height=500,
            placeholder="<strong>The answers will be evaluted here</strong><br>You can also ask Me Anything",
        )
        chatbot_input = gr.Textbox(
            submit_btn=True, placeholder="Type a message...", show_label=False
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
            triggers=[question_submit_button.click, answer.submit],
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
