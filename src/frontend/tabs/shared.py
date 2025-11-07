"""Shared functions for the frontend tabs"""

import logging

import gradio as gr
import numpy as np
from llama_index.core.workflow import Context

import prompts
from backend.audio import generate_audio
from backend.chatbot.chatbot_workflow import ChatBotWorkfLow
from backend.events import AudioStreamEvent, ChatBotStartEvent, LLMProgressEvent
from backend.question_generator.base import QuestionBuffer, QuestionGenerator
from util import const

logger = logging.getLogger(__name__)


def get_question_generator(state: dict, model: str):
    """Creates a Question Generator. Creates a new Question Buffer if none exists in the state. Otherwise reuses the existing one."""
    if state.keys().__contains__("question_buffer"):
        buffer = state["question_buffer"]
    else:
        buffer = QuestionBuffer()
        state["question_buffer"] = buffer

    question_generator = QuestionGenerator(model, buffer=buffer)

    return state, question_generator


def create_reading_comprehension_user_message(
    topic: str, text: str, question: str, answer: str
):
    """Creates the user message for reading comprehension verification."""
    return prompts.READING_COMPREHENSION_REQUEST_PROMPT.format(
        topic=topic, text=text, question=question, answer=answer
    )


async def chat(
    message: str,
    history: list,
    is_stream: bool,
    audio_output: bool,
    model: str,
    embedding_model: str,
    search_engine: str,
    prompt: str | dict[str, str] | None = None,
):
    """The main chat function used for communicating with the chatbot workflow."""
    try:
        workflow = ChatBotWorkfLow(
            prompt=prompt,
            timeout=None,
        )

        ctx = Context(workflow)
        await ctx.store.set(const.IS_STREAM, is_stream)
        await ctx.store.set(const.AUDIO_OUTPUT, audio_output)
        await ctx.store.set(const.MODEL, model)
        await ctx.store.set(const.EMBEDDING_MODEL, embedding_model)
        await ctx.store.set(const.SEARCH_ENGINE, search_engine)
        response = ""

        start_event = ChatBotStartEvent(message=message, history=history)

        handler = workflow.run(start_event=start_event, ctx=ctx)

        async for event in handler.stream_events():
            if isinstance(event, LLMProgressEvent):
                response += event.response
                yield response

            if isinstance(event, AudioStreamEvent):
                yield [
                    response,
                    gr.Audio(
                        event.audio_chunk,
                        type="numpy",
                        streaming=True,
                        autoplay=True,
                        interactive=False,
                    ),
                ]
    except Exception:
        gr.Error("There was an error. Please try again.")


async def assistant_chat(
    message: str,
    history: list,
    is_stream: bool,
    audio_output: bool,
    model: str,
    embedding_model: str,
    search_engine: str,
):
    """Chat function for the assistant chatbot."""
    async for event in chat(
        message,
        history,
        is_stream,
        audio_output,
        model,
        embedding_model,
        search_engine,
        None,
    ):
        yield event


async def reading_comprehension_chat(
    message: str,
    history: list,
    is_stream: bool,
    audio_output: bool,
    model: str,
    embedding_model: str,
    search_engine: str,
):
    """Chat function for the reading comprehension chatbot. This chat will be used to evaluate the user's answers."""
    async for event in reading_comprehension_verifier(
        message,
        None,
        None,
        None,
        history,
        is_stream,
        audio_output,
        model,
        embedding_model,
        search_engine,
    ):
        yield event, ""


async def reading_comprehension_verifier(
    topic: str,
    text: str,
    question: str,
    answer: str,
    history: list,
    is_stream: bool,
    audio_output: bool,
    model: str,
    embedding_model: str,
    search_engine: str,
):
    chat_prompts = {
        "function": prompts.READING_COMPREHENSION_FUNCTION_CHATBOT_PROMPT,
        "react": prompts.READING_COMPREHENSION_REACT_CHATBOT_PROMPT,
    }

    if answer is None:
        user_message = topic
    else:
        user_message = create_reading_comprehension_user_message(
            topic, text, question, answer
        )

    # Update shown history
    current_history = history.copy()
    current_history.append({"role": "user", "content": user_message})
    yield current_history

    async for event in chat(
        user_message,
        history,
        is_stream,
        audio_output,
        model,
        embedding_model,
        search_engine,
        chat_prompts,
    ):
        if isinstance(event, str):
            current_history.append({"role": "assistant", "content": event})

            yield current_history
        else:
            logger.warning(
                "Translation chat got an event which is currently not supported"
            )


def process_select(state, c1, c2, c3, c4):
    """Processes the selection of options in a multiple-choice question. Ensures only one option is selected at a time."""
    options = [c1, c2, c3, c4]

    assert options.count(True) <= 2, "The option select broke"

    if options.count(True) > 1:
        options[state["selected_option"]] = False

    for i, o in enumerate(options):
        if o:
            state["selected_option"] = i

    return state, options[0], options[1], options[2], options[3]


def process_unselect(state, c1, c2, c3, c4):
    """Processes the unselection of options in a multiple-choice question."""
    options = [c1, c2, c3, c4]

    if options.count(False) == len(options):
        state["selected_option"] = None

    return state


def verify_input(language: str, language_proficiency: str, difficulty: str):
    """Verifies that the input parameters are valid. Raises an error if any of the parameters are invalid."""
    if language == "":
        raise gr.Error("No language was input. Please add one in the right sidebar.")
    if language_proficiency == "":
        raise gr.Error(
            "No language profiency was input. Please add one in the right sidebar."
        )
    if difficulty == "":
        raise gr.Error("No difficulty was input. Please add one in the right sidebar.")


async def get_audio(tts_provider: str, language: str, *args: tuple[str]):
    """Generates audio for the given texts using the specified TTS provider and language."""
    complete_audio = np.array([])

    for text in args:
        for sr, audio_np in generate_audio(tts_provider, text, language):
            logger.info(
                f"Generated audio chunk with shape {audio_np.shape} and sample rate {sr}"
            )
            complete_audio = np.concatenate((complete_audio, audio_np))

        complete_audio = np.concatenate((complete_audio, np.zeros(2 * sr)))

    yield (sr, complete_audio)


def clear():
    return None
