import gradio as gr

from frontend.gui import create_gui


def vote(data: gr.LikeData):
    if data.liked:
        print("You upvoted this response: " + data.value["value"])
    else:
        print("You downvoted this response: " + data.value["value"])


if __name__ == "__main__":
    demo = create_gui()

    demo.launch()
