from frontend.gui import create_gui

import logging

import mlflow

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    mlflow.llama_index.autolog()
    mlflow.set_experiment("LLA-Agent")

    demo = create_gui()

    demo.launch()
