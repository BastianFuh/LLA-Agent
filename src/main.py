import grequests
from frontend.gui import create_gui

import logging

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

from phoenix.otel import register

import gradio as gr

if __name__ == "__main__":
    tracer_provider = register(project_name="LLA-Agent", batch=True, verbose=False)
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

    logging.basicConfig(level=logging.INFO)
    with gr.Blocks(fill_height=True) as demo:
        create_gui()

    demo.launch()
