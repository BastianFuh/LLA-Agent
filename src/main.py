# grequest needs to be up here
# If not it breaks some request for some reason...
# Might be fixed by removing it in the file it is used
from pathlib import Path
import grequests
import logging

import gradio as gr
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register

from frontend.gui import create_gui

if __name__ == "__main__":
    css_path = (
        Path(__file__).parents[0] / Path("..") / Path("resource") / Path("style.css")
    )

    with open(css_path, "r") as f:
        custom_css = f.read()

    tracer_provider = register(project_name="LLA-Agent", batch=True, verbose=False)
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

    logging.basicConfig(level=logging.INFO)
    with gr.Blocks(
        theme=gr.themes.Default(text_size=gr.themes.sizes.text_lg),
        fill_height=True,
        css=custom_css,
    ) as demo:
        create_gui()

    demo.launch()
