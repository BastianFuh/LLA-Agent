import logging
import random
from textwrap import TextWrapper, wrap

import gradio as gr
import numpy as np
from elevenlabs import ElevenLabs
from kokoro import KPipeline
from llama_index.core.workflow import Context
from openai import AsyncOpenAI

import prompts
from util import const
from util.audio import get_voice_samples
from workflow.chatbot.chatbot_workflow import ChatBotWorkfLow
from workflow.events import AudioStreamEvent, ChatBotStartEvent, LLMProgressEvent
from workflow.question_generator import tools as QGT
from workflow.question_generator.base import QuestionBuffer, QuestionGenerator

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
    state: gr.State,
    model: str,
    language: str,
    language_proficiency: str,
    difficulty: str,
    additional_information: str,
):
    _verify_input(language, language_proficiency, difficulty)

    state, question_generator = _get_question_generator(state, model)

    async for question_data in question_generator.generate_reading_comprehension(
        language, language_proficiency, difficulty, additional_information
    ):
        topic = f"{question_data[QGT.READING_COMPREHENSION_TOPIC]}"
        text = f"{question_data[QGT.READING_COMPREHENSION_TEXT]}"
        question = f"{question_data[QGT.READING_COMPREHENSION_QUESTION]}"

        wrapped_text = text_wrapper.fill(text)

        yield (
            state,
            gr.Textbox(value=topic),
            gr.TextArea(value=wrapped_text),
            gr.Textbox(value=question),
            gr.Textbox(value="", info=""),
        )

    yield gr.skip()


async def create_audio(tts_provider: str, language: str, *args: tuple[str]):
    output_text = args[0]
    for arg in args[1:]:
        # The added dots should results in breaks during speaking
        if tts_provider == const.TTS_KOKORO:
            output_text += ". " + arg
        else:
            output_text += arg

    # Clear current audio
    yield gr.Audio(value=None, streaming=False)

    output_text = output_text.replace("\n", "")

    sr = 24000

    if tts_provider == const.TTS_KOKORO:
        yield gr.Audio(value=None, streaming=True)
        pipeline = KPipeline(lang_code="j", repo_id="hexgrad/Kokoro-82M")
        generator = pipeline(
            output_text,
            voice="jf_alpha",  # <= change voice here
            speed=1,
            split_pattern=r"(\.|。|!|\?|！|？|、)",
        )

        for gs, ps, audio in generator:
            yield (sr, audio.numpy())

    if tts_provider == const.TTS_ELEVENLABS:
        from io import BytesIO

        client = ElevenLabs()

        audio = client.text_to_speech.convert(
            text=output_text,
            voice_id="GxxMAMfQkDlnqjpzjLHH",
            model_id="eleven_flash_v2_5",
            output_format="pcm_24000",
        )

        stream = BytesIO()

        for chunk in audio:
            if chunk:
                stream.write(chunk)
        stream.seek(0)

        # return_audio = np.frombuffer(audio, dtype=np.int16)
        yield (sr, np.frombuffer(stream.getbuffer(), dtype=np.int16))

    if tts_provider == const.TTS_OPENAI:
        voices = [
            "alloy",
            "ballad",
            "coral",
            "echo",
            "fable",
            "onyx",
            "nova",
            "sage",
            "shimmer",
            "verse",
        ]

        instructions = f"""Language: Speak in {language} with a standard accent.

                    Voice Affect: Composed; project authority and confidence.

                    Tone: Sincere and authoritative—express genuine apology while conveying competence.

                    Pacing: Steady and moderate.

                    Pronunciation: Clear and precise."""

        client = AsyncOpenAI()
        response = await client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=voices[random.randint(0, len(voices) - 1)],
            input=output_text,
            instructions=instructions,
            response_format="pcm",
        )
        return_audio = np.frombuffer(response.content, dtype=np.int16)

        yield (sr, return_audio)

    if tts_provider == const.TTS_FISH_AUDIO:
        import io
        import wave

        from fish_audio_sdk import ReferenceAudio, Session, TTSRequest

        client = Session(apikey="LOCAL", base_url="http://127.0.0.1:8080")

        sr = 21000

        wave_file = io.BytesIO()

        voices = get_voice_samples(language)

        if len(voices) != 0:
            # Select a random voice from the list
            voice = voices[random.randint(0, len(voices) - 1)]

            # Prepare the voice reference

            audio_path = voice["audio_file"]

            # Read wav file to bytes
            with open(audio_path, "rb") as f:
                voice_audio = f.read()

            references = [
                ReferenceAudio(
                    audio=voice_audio,
                    text=voice["text"],
                )
            ]
        else:
            references = []

        for chunk in client.tts(
            TTSRequest(text=output_text, format="wav", references=references)
        ):
            wave_file.write(chunk)
        wave_file.seek(0)

        with wave.open(wave_file, "rb") as wav_file:
            num_channels = wav_file.getnchannels()
            framerate = wav_file.getframerate()
            num_frames = wav_file.getnframes()

            audio_data = wav_file.readframes(num_frames)

            audio_np = np.frombuffer(audio_data, dtype=np.int16)

            # If stereo (2 channels), reshape to [num_frames, 2] array
            if num_channels == 2:
                audio_np = audio_np.reshape(-1, 2)
        sr = framerate
        yield (framerate, audio_np)
    # Flushes the output, because if audio is streamed, it sometimes does not realize only one message
    # being yielded
    yield (sr, np.zeros(2400))


def clear():
    return None
