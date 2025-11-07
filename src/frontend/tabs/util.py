import gradio as gr

import frontend.tabs.shared as F
from util.transcription import AudioTranscriber


def handle_audio_output_while_not_recording(state: dict) -> str | dict:
    """Handle audio output while not recording."""

    if state["initialized"]:
        return gr.skip()
    else:
        state["initialized"] = True
        return ""


# Transcriber instance to be used for textboxes with audio input
# One instance is shared across the application to avoid multiple instances at the same time
transcriber = AudioTranscriber(handle_audio_output_while_not_recording)


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
