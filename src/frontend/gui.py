import os

import gradio as gr
from gradio_toggle import Toggle
from ollama import show

import frontend.functions as F
from util.const import (
    OPTION_SEARCH_ENGINE,
    TTS_CHATTERBOX,
    TTS_ELEVENLABS,
    TTS_FISH_AUDIO,
    TTS_KOKORO,
    TTS_OPENAI,
)
from util.model import get_embedding_models, get_llm_models
from util.transcription import AudioTranscriber


def handle_audio_output_while_not_recording(state: dict) -> str | dict:
    """Handle audio output while not recording."""

    if state["initialized"]:
        return gr.skip()
    else:
        state["initialized"] = True
        return ""


transcriber = AudioTranscriber(handle_audio_output_while_not_recording)


def create_gui() -> gr.Blocks:
    default_browser_state = {
        "language": "",
        "language_proficiency": "",
        "difficulty": "",
        "additional_information": "",
        "option_model": list(get_llm_models().keys())[0],
        "option_embedding_model": list(get_embedding_models().keys())[0],
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

    with gr.Sidebar(label="Settings", position="right"):
        with gr.Group():
            language = create_text_input(browser_state, False, "language", "Language")
            language_proficiency = create_text_input(
                browser_state, False, "language_proficiency", "Language Proficiency"
            )
            difficulty = create_text_input(
                browser_state, False, "difficulty", "Difficulty"
            )
            additional_information = create_text_input(
                browser_state, True, "additional_information", "Additional Information"
            )

    with gr.Tabs(selected=1):
        with gr.Tab("Options", id=0, scale=1):
            options = list(get_llm_models().keys())
            model = create_dropdown_input(
                browser_state, options, "option_model", "Chatbot Model"
            )

            embedding_model = create_dropdown_input(
                browser_state,
                list(get_embedding_models().keys()),
                "option_embedding_model",
                "Embedding Model",
            )

            search_engine = create_dropdown_input(
                browser_state,
                OPTION_SEARCH_ENGINE,
                "option_search_engine",
                "Search Engine",
            )

            tts_provider = create_dropdown_input(
                browser_state,
                [
                    TTS_KOKORO,
                    TTS_ELEVENLABS,
                    TTS_OPENAI,
                    TTS_FISH_AUDIO,
                    TTS_CHATTERBOX,
                ],
                "tts_provider",
                "TTS Provider",
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

        with gr.Tab("Conversation Exercise", id=1, scale=1):
            create_conversation_tab(
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

        with gr.Tab("Multiple Choice Fill-in-the-blank", id=2, scale=1):
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

        with gr.Tab("Constructed response fill-in-the-blank", id=3, scale=1):
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
                tts_provider,
                language,
                language_proficiency,
                difficulty,
                additional_information,
            )

        with gr.Tab("Comprehension - Monologue", id=5, scale=1):
            create_reading_comprehension_question(
                browser_state,
                create_state,
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
            )

        with gr.Tab("Comprehension - Dialogue", id=6, scale=1):
            create_listening_comprehension_question(
                browser_state,
                create_state,
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
            )


def create_conversation_tab(
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
    chatbot = gr.Chatbot(
        type="messages",
        scale=1,
        placeholder="<strong>Your Personal Language Learning Assistant</strong><br>Ask Me Anything",
    )
    gr.ChatInterface(
        fn=F.conversation_chat,
        type="messages",
        chatbot=chatbot,
        additional_inputs=[
            is_stream,
            audio_output,
            model,
            embedding_model,
            search_engine,
            language,
            language_proficiency,
            difficulty,
            additional_information,
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


def create_translation_question(
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


def create_reading_comprehension_question(
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


def create_listening_comprehension_question(
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


def create_audio_output(tts_provider, language, *text_input_elements):
    with gr.Group():
        audio_player = gr.Audio(interactive=False, scale=4, type="numpy")
        generate_button = gr.Button("Generate Audio", scale=1)

    generate_button.click(
        fn=F.clear,
        outputs=[audio_player],
    ).then(
        fn=F.get_audio,
        inputs=[tts_provider, language] + list(text_input_elements),
        outputs=[audio_player],
    )

    return audio_player


def create_textbox_with_audio_input(**text_box_kargs) -> gr.Textbox:
    audio_input_state = gr.State({"initialized": False})
    with gr.Group():
        text_box = gr.Textbox(
            interactive=True,
            value=transcriber.get_text,
            inputs=[audio_input_state],
            every=0.1,
            scale=2,
            **text_box_kargs,
        )

        audio_input = gr.Audio(
            label="Audio Transcription for Textbox",
            sources="microphone",
            type="numpy",
            streaming=True,
            scale=1,
            waveform_options=gr.WaveformOptions(show_recording_waveform=False),
        )

        audio_input.start_recording(transcriber.start_recording, trigger_mode="once")
        audio_input.stop_recording(transcriber.stop_recording, trigger_mode="once")

        audio_input.stream(
            fn=transcriber.feed_audio,
            inputs=[audio_input],
        )

    return text_box


def create_chatbot(
    placeholder: str = "<strong>Your Personal Language Learning Assistant</strong><br>Ask Me Anything",
) -> gr.Chatbot:
    return gr.Chatbot(
        type="messages",
        show_copy_button=True,
        scale=1,
        placeholder=placeholder,
    )
