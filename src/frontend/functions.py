import logging
from ast import mod
from textwrap import TextWrapper

import gradio as gr
import numpy as np
from llama_index.core.workflow import Context

import prompts
from backend.audio import generate_audio, get_random_voice_id_for_provider
from backend.chatbot.chatbot_workflow import ChatBotWorkfLow
from backend.events import AudioStreamEvent, ChatBotStartEvent, LLMProgressEvent
from backend.question_generator import tools as QGT
from backend.question_generator.base import QuestionBuffer, QuestionGenerator
from util import const

text_wrapper = TextWrapper(width=37, expand_tabs=False, replace_whitespace=False)


def _get_question_generator(state: dict, model: str):
    if state.keys().__contains__("question_buffer"):
        buffer = state["question_buffer"]
    else:
        buffer = QuestionBuffer()
        state["question_buffer"] = buffer

    question_generator = QuestionGenerator(model, buffer=buffer)

    return state, question_generator


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


async def conversation_chat(
    message: str,
    history: list,
    is_stream: bool,
    audio_output: bool,
    model: str,
    embedding_model: str,
    search_engine: str,
    language: str,
    language_proficiency: str,
    difficulty: str,
    additional_information: str,
):
    _verify_input(language, language_proficiency, difficulty)

    prompt = prompts.CONVERSATION_BOT_FUNCTION_PROMPT

    options = {
        "language": language,
        "language_proficiency": language_proficiency,
        "difficulty": difficulty,
        "additional_information": additional_information,
    }

    prompt = prompt.format(**options)

    async for event in chat(
        message,
        history,
        is_stream,
        audio_output,
        model,
        embedding_model,
        search_engine,
        prompt,
    ):
        yield event


async def assistant_chat(
    message: str,
    history: list,
    is_stream: bool,
    audio_output: bool,
    model: str,
    embedding_model: str,
    search_engine: str,
):
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


def _create_translation_user_message(original: str, answer: str):
    return prompts.QUESTION_GENERATOR_TRANSLATION_REQUEST_PROMPT.format(
        original_text=original, translation=answer
    )


def _create_reading_comprehension_user_message(
    topic: str, text: str, question: str, answer: str
):
    return prompts.READING_COMPREHENSION_REQUEST_PROMPT.format(
        topic=topic, text=text, question=question, answer=answer
    )


async def translation_verifier_chat(
    message: str,
    history: list,
    is_stream: bool,
    audio_output: bool,
    model: str,
    embedding_model: str,
    search_engine: str,
):
    async for event in translation_verifier(
        None,
        history,
        message,
        is_stream,
        audio_output,
        model,
        embedding_model,
        search_engine,
    ):
        yield event, ""


async def reading_comprehension_chat(
    message: str,
    history: list,
    is_stream: bool,
    audio_output: bool,
    model: str,
    embedding_model: str,
    search_engine: str,
):
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


async def translation_verifier(
    answer: str | None,
    history: list,
    original: str | None,
    is_stream: bool,
    audio_output: bool,
    model: str,
    embedding_model: str,
    search_engine: str,
):
    chat_prompts = {
        "function": prompts.QUESTION_GENERATOR_TRANSLATION_FUNCTION_CHATBOT_PROMPT,
        "react": prompts.QUESTION_GENERATOR_TRANSLATION_REACT_CHATBOT_PROMPT,
    }

    if answer is None:
        user_message = original
    else:
        user_message = _create_translation_user_message(original, answer)

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
            logging.warning(
                "Translation chat got an event which is currently not supported"
            )


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
        user_message = _create_reading_comprehension_user_message(
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
            logging.warning(
                "Translation chat got an event which is currently not supported"
            )


async def listening_comprehension_verifier(
    state: dict,
    topic: str,
    question: str,
    answer: str,
    history: list,
    is_stream: bool,
    audio_output: bool,
    model: str,
    embedding_model: str,
    search_engine: str,
):
    data = state["listening_comprehension_data"]

    text = data[QGT.LISTENING_COMPREHENSION_TEXT]

    complete_text = "\n".join(
        [f"{text_segment['speaker']}: {text_segment['text']}" for text_segment in text]
    )

    chat_prompts = {
        "function": prompts.READING_COMPREHENSION_FUNCTION_CHATBOT_PROMPT,
        "react": prompts.READING_COMPREHENSION_REACT_CHATBOT_PROMPT,
    }

    if answer is None:
        user_message = topic
    else:
        user_message = _create_reading_comprehension_user_message(
            topic, complete_text, question, answer
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
            logging.warning(
                "Translation chat got an event which is currently not supported"
            )


def process_select(state, c1, c2, c3, c4):
    options = [c1, c2, c3, c4]

    assert options.count(True) <= 2, "The option select broke"

    if options.count(True) > 1:
        options[state["selected_option"]] = False

    for i, o in enumerate(options):
        if o:
            state["selected_option"] = i

    return state, options[0], options[1], options[2], options[3]


def process_unselect(state, c1, c2, c3, c4):
    options = [c1, c2, c3, c4]

    if options.count(False) == len(options):
        state["selected_option"] = None

    return state


def _verify_input(language: str, language_proficiency: str, difficulty: str):
    if language == "":
        raise gr.Error("No language was input. Please add one in the right sidebar.")
    if language_proficiency == "":
        raise gr.Error(
            "No language profiency was input. Please add one in the right sidebar."
        )
    if difficulty == "":
        raise gr.Error("No difficulty was input. Please add one in the right sidebar.")


async def create_multiple_choice_questions(
    state: gr.State,
    model: str,
    language: str,
    language_proficiency: str,
    difficulty: str,
    additional_information: str,
):
    _verify_input(language, language_proficiency, difficulty)

    state, question_generator = _get_question_generator(state, model)

    async for question_data in question_generator.generate_multiple_choice(
        language, language_proficiency, difficulty, additional_information
    ):
        state[QGT.QUESTION_ANSWER_INDEX] = question_data[QGT.QUESTION_ANSWER_INDEX]

        options = question_data[QGT.QUESTION_OPTIONS]

        question_text = f"{question_data[QGT.QUESTION_TEXT]}"

        yield (
            state,
            gr.Textbox(value=question_text, info=question_data[QGT.QUESTION_HINT]),
            gr.Checkbox(label=options[0], value=False, info=""),
            gr.Checkbox(label=options[1], value=False, info=""),
            gr.Checkbox(label=options[2], value=False, info=""),
            gr.Checkbox(label=options[3], value=False, info=""),
        )

    yield gr.skip()


async def verify_multiple_choice_question(state: dict, c1, c2, c3, c4):
    options = [c1, c2, c3, c4]

    if not state.__contains__(QGT.QUESTION_ANSWER_INDEX):
        gr.Info("Please generate a question first.")
        return gr.skip()

    correct_answer = state[QGT.QUESTION_ANSWER_INDEX]

    try:
        selected_answer = options.index(True)
    except ValueError:
        gr.Info("Please select a option")
        return gr.skip()

    updates = len(options) * [gr.Checkbox(info="")]

    if options[correct_answer]:
        updates[correct_answer] = gr.Checkbox(info="Correct Answer")
    else:
        updates[selected_answer] = gr.Checkbox(info="Wrong Answer")

    return updates


async def create_free_text_question(
    state: gr.State,
    model: str,
    language: str,
    language_proficiency: str,
    difficulty: str,
    additional_information: str,
):
    _verify_input(language, language_proficiency, difficulty)

    state, question_generator = _get_question_generator(state, model)

    async for question_data in question_generator.generate_free_text(
        language, language_proficiency, difficulty, additional_information
    ):
        state[QGT.QUESTION_ANSWER] = question_data[QGT.QUESTION_ANSWER]

        question_text = f"{question_data[QGT.QUESTION_TEXT]}"

        yield (
            state,
            gr.Textbox(value=question_text, info=question_data[QGT.QUESTION_HINT]),
            gr.Textbox(value="", info=""),
        )

    yield gr.skip()


async def verify_free_text_question(state: dict, answer: str):
    if not state.__contains__(QGT.QUESTION_ANSWER):
        gr.Info("Please generate a question first.")
        return gr.skip()

    correct_answer = state[QGT.QUESTION_ANSWER]

    if correct_answer == answer:
        update_answer_field = gr.Textbox(info="Correct Answer")
    else:
        update_answer_field = gr.Textbox(info="Wrong Answer")

    return update_answer_field


async def show_free_text_answer(state: dict):
    if not state.__contains__(QGT.QUESTION_ANSWER):
        gr.Info("Please generate a question first.")
        return gr.skip()

    return gr.Textbox(info=state[QGT.QUESTION_ANSWER])


async def create_translation_question(
    state: gr.State,
    model: str,
    language: str,
    language_proficiency: str,
    difficulty: str,
    additional_information: str,
):
    _verify_input(language, language_proficiency, difficulty)

    state, question_generator = _get_question_generator(state, model)

    async for question_data in question_generator.generate_translation_question(
        language, language_proficiency, difficulty, additional_information
    ):
        question_text = f"{question_data[QGT.QUESTION_BASE_TEXT]}"

        yield (
            state,
            gr.Textbox(value=question_text),
            gr.Textbox(value="", info=""),
        )

    yield gr.skip()


async def create_reading_comprehension_question(
    state: dict,
    model: str,
    language: str,
    language_proficiency: str,
    difficulty: str,
    additional_information: str,
    mode_switch: bool,
    tts_provider: str,
):
    _verify_input(language, language_proficiency, difficulty)

    state, question_generator = _get_question_generator(state, model)

    yield (
        gr.skip(),
        gr.Textbox(value=""),
        gr.TextArea(value=""),
        gr.Textbox(value=""),
        gr.Textbox(value="", info=""),
        gr.skip(),
    )

    async for question_data in question_generator.generate_reading_comprehension(
        language,
        language_proficiency,
        difficulty,
        additional_information,
        mode_switch,
        tts_provider,
    ):
        topic = f"{question_data[QGT.READING_COMPREHENSION_TOPIC]}"
        text = f"{question_data[QGT.READING_COMPREHENSION_TEXT]}"
        question = f"{question_data[QGT.READING_COMPREHENSION_QUESTION]}"

        wrapped_text = text_wrapper.fill(text)

        if mode_switch:
            yield (
                state,
                gr.Textbox(value=topic, visible=False),
                gr.TextArea(value=wrapped_text, visible=False),
                gr.Textbox(value=question, visible=False),
                gr.Textbox(value="", info=""),
                question_data[QGT.AUDIO_DATA],
            )
        else:
            yield (
                state,
                gr.Textbox(value=topic),
                gr.TextArea(value=wrapped_text),
                gr.Textbox(value=question),
                gr.Textbox(value="", info=""),
                gr.skip(),
            )


async def create_listening_comprehension_question(
    state: dict,
    model: str,
    language: str,
    language_proficiency: str,
    difficulty: str,
    additional_information: str,
    mode_switch: bool,
    tts_provider: str,
):
    _verify_input(language, language_proficiency, difficulty)

    state, question_generator = _get_question_generator(state, model)

    yield (
        gr.skip(),
        gr.Textbox(value=""),
        gr.Textbox(value=""),
        gr.Textbox(value=""),
        gr.Chatbot(value=[], type="messages"),
        gr.Textbox(value=""),
        gr.Textbox(value="", info=""),
        gr.skip(),
    )

    async for question_data in question_generator.generate_listening_comprehension(
        language,
        language_proficiency,
        difficulty,
        additional_information,
        mode_switch,
        tts_provider,
    ):
        state["listening_comprehension_data"] = question_data
        topic = f"{question_data[QGT.LISTENING_COMPREHENSION_TOPIC]}"
        speakers = question_data[QGT.LISTENING_COMPREHENSION_SPEAKERS]
        text = question_data[QGT.LISTENING_COMPREHENSION_TEXT]
        question = f"{question_data[QGT.LISTENING_COMPREHENSION_QUESTION]}"

        role_mapping = ["assistant", "user"]

        text_messages = [
            gr.ChatMessage(role=role_mapping[i % 2], content=text_segment["text"])
            for i, text_segment in enumerate(text)
        ]

        if mode_switch:
            yield (
                state,
                gr.Textbox(value=topic),
                gr.Textbox(value=speakers[0]),
                gr.Textbox(value=speakers[1]),
                gr.skip(),
                gr.Textbox(value=question),
                gr.Textbox(value="", info=""),
                question_data[QGT.AUDIO_DATA],
            )
        else:
            yield (
                state,
                gr.Textbox(value=topic),
                gr.Textbox(value=speakers[0]),
                gr.Textbox(value=speakers[1]),
                text_messages,
                gr.Textbox(value=question),
                gr.Textbox(value="", info=""),
                gr.skip(),
            )


async def show_comprehension_text(state, topic: str, text: str, question: str):
    yield (
        state,
        gr.Textbox(value=topic, visible=True),
        gr.TextArea(value=text, visible=True),
        gr.Textbox(value=question, visible=True),
    )


async def show_listening_comprehension_text(state):
    role_mapping = ["assistant", "user"]

    data = state["listening_comprehension_data"]

    text = data[QGT.LISTENING_COMPREHENSION_TEXT]

    text_messages = [
        gr.ChatMessage(role=role_mapping[i % 2], content=text_segment["text"])
        for i, text_segment in enumerate(text)
    ]

    yield (
        state,
        text_messages,
    )


async def get_audio(tts_provider: str, language: str, *args: tuple[str]):
    complete_audio = np.array([])

    for text in args:
        for sr, audio_np in generate_audio(tts_provider, text, language):
            logging.info(
                f"Generated audio chunk with shape {audio_np.shape} and sample rate {sr}"
            )
            complete_audio = np.concatenate((complete_audio, audio_np))

        complete_audio = np.concatenate((complete_audio, np.zeros(2 * sr)))

    yield (sr, complete_audio)


def clear():
    return None
