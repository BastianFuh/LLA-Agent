import logging
from pathlib import Path

import gradio as gr
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register

from backend.audio import init_fish_audio_voice_samples
from util.model import init_models

logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
root_logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    init_fish_audio_voice_samples()
    init_models()

    from frontend.gui import create_gui

    css_path = (
        Path(__file__).parents[0] / Path("..") / Path("resource") / Path("style.css")
    )

    with open(css_path, "r") as f:
        custom_css = f.read()

    tracer_provider = register(project_name="LLA-Agent", batch=True, verbose=False)
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

    # theme: gr.themes.Base = gr.themes.ThemeClass().from_hub("allenai/gradio-theme")
    theme = gr.themes.Default(text_size=gr.themes.sizes.text_lg)

    with gr.Blocks(
        theme=theme,
        fill_height=True,
        css=custom_css,
        title="LLA-Agent",
    ) as demo:
        create_gui()

    demo.launch()
