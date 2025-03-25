from frontend.gui import create_gui

import logging

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    demo = create_gui()

    demo.launch()
