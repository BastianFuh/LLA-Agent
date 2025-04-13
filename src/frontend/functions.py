from llama_index.core.workflow import Context

from workflow.events import LLMProgressEvent

from workflow.chatbot.chatbot_workflow import ChatBotWorkfLow
from workflow.events import ChatBotStartEvent, AudioStreamEvent
from workflow.question_generator.base import QuestionGenerator
from workflow.question_generator import tools as QGT

from util import const

import gradio as gr

import prompts

import logging


async def chat(
    message: str,
    history: dict,
    is_stream: bool,
    audio_output: bool,
    model: str,
    embedding_model: str,
    search_engine: str,
    prompt: str | dict[str, str] = None,
):
    try:
        workflow = ChatBotWorkfLow(
            prompt=prompt,
            timeout=None,
        )

        ctx = Context(workflow)
        await ctx.set(const.IS_STREAM, is_stream)
        await ctx.set(const.AUDIO_OUTPUT, audio_output)
        await ctx.set(const.MODEL, model)
        await ctx.set(const.EMBEDDING_MODEL, embedding_model)
        await ctx.set(const.SEARCH_ENGINE, search_engine)
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


async def basic_chat(
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
        message,
        history,
        None,
        is_stream,
        audio_output,
        model,
        embedding_model,
        search_engine,
    ):
        yield event


async def translation_verifier(
    answer: str,
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
            history.append({"role": "assistant", "content": event})

            yield history
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


def verify_input(language: str, language_proficiency: str, difficulty: str):
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
    verify_input(language, language_proficiency, difficulty)

    question_generator = QuestionGenerator(model)

    question_data = await question_generator.generate_multiple_choice(
        language, language_proficiency, difficulty, additional_information
    )

    state[QGT.QUESTION_ANSWER_INDEX] = question_data[QGT.QUESTION_ANSWER_INDEX]

    options = question_data[QGT.QUESTION_OPTIONS]

    question_text = f"{question_data[QGT.QUESTION_TEXT]}"

    return (
        gr.Textbox(value=question_text, info=question_data[QGT.QUESTION_HINT]),
        gr.Checkbox(label=options[0], value=False, info=""),
        gr.Checkbox(label=options[1], value=False, info=""),
        gr.Checkbox(label=options[2], value=False, info=""),
        gr.Checkbox(label=options[3], value=False, info=""),
    )


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
    verify_input(language, language_proficiency, difficulty)

    question_generator = QuestionGenerator(model)

    question_data = await question_generator.generate_free_text(
        language, language_proficiency, difficulty, additional_information
    )

    state[QGT.QUESTION_ANSWER] = question_data[QGT.QUESTION_ANSWER]

    question_text = f"{question_data[QGT.QUESTION_TEXT]}"

    return (
        gr.Textbox(value=question_text, info=question_data[QGT.QUESTION_HINT]),
        gr.Textbox(value="", info=""),
    )


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
    verify_input(language, language_proficiency, difficulty)

    question_generator = QuestionGenerator(model)

    question_data = await question_generator.generate_translation_question(
        language, language_proficiency, difficulty, additional_information
    )

    question_text = f"{question_data[QGT.QUESTION_BASE_TEXT]}"

    return (
        gr.Textbox(value=question_text),
        gr.Textbox(value="", info=""),
    )


def append_to_chatbot_history(history: list, original: str, answer: str) -> list[dict]:
    if answer is None:
        message = original
    else:
        message = _create_translation_user_message(original, answer)
    history.append({"role": "user", "content": message})
    return history, ""
