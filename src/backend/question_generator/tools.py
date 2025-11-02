import logging
import random

import numpy as np
from llama_index.core.workflow import Context

import backend.question_generator as question_generator
from backend.audio import generate_audio, get_random_voice_id_for_provider

logger = logging.getLogger(__name__)

QUESTION_BASE_TEXT = "question_base_text"
QUESTION_TEXT = "question_text"
QUESTION_OPTIONS = "question_options"
QUESTION_ANSWER = "question_answer"
QUESTION_ANSWER_INDEX = "question_answer_index"
QUESTION_HINT = "question_hint"

READING_COMPREHENSION_TOPIC = "reading_comprehension_topic"
READING_COMPREHENSION_TEXT = "reading_comprehension_text"
READING_COMPREHENSION_QUESTION = "reading_comprehension_question"

LISTENING_COMPREHENSION_TOPIC = "listening_comprehension_topic"
LISTENING_COMPREHENSION_SPEAKERS = "listening_comprehension_speakers"
LISTENING_COMPREHENSION_TEXT = "listening_comprehension_text"
LISTENING_COMPREHENSION_QUESTION = "listening_comprehension_question"

AUDIO_DATA = "audio_data"

_CREATE_QUESTION_BASE_TEXT_INSTRUCTION = """
You have started the question generation with this text:

```
{text}
```

Every following step MUST ensure that the question is about {focus}.

"""

_CREATE_COMPREHENSION_BASE_TEXT_INSTRUCTION = """
You have started the generation with this text:

```
{text}
```

"""


async def finish(context: Context) -> str:
    """This function is used to finish the current process or operation.
    This must be called at the end of every process.

    Returns:
        str: Return information.
    """
    question_type = await context.store.get("question_type")

    match question_type:
        case question_generator.READING_COMPREHENSION:
            topic = await context.store.get(READING_COMPREHENSION_TOPIC, None)
            text = await context.store.get(READING_COMPREHENSION_TEXT, None)
            question = await context.store.get(READING_COMPREHENSION_QUESTION, None)

            if topic is None:
                return "You have not provided a topic for the reading comprehension. Please ensure that you have provided all the necessary information."

            if text is None:
                return "You have not provided a text for the reading comprehension. Please ensure that you have provided all the necessary information."

            if question is None:
                return "You have not provided a question for the reading comprehension. Please ensure that you have provided all the necessary information."

            extra_parameters = await context.store.get("extra_parameters")

            mode_switch = extra_parameters["mode_switch"]

            if mode_switch:
                complete_audio = np.array([])
                tts_provider = extra_parameters["tts_provider"]
                language = extra_parameters["language"]

                general_speaker = get_random_voice_id_for_provider(
                    tts_provider, language
                )
                text_speaker = get_random_voice_id_for_provider(
                    tts_provider, language, exclude_ids=[general_speaker]
                )

                for sr, audio_np in generate_audio(
                    tts_provider, topic, language, general_speaker
                ):
                    complete_audio = np.concatenate((complete_audio, audio_np))

                complete_audio = np.concatenate((complete_audio, np.zeros(2 * sr)))

                for sr, audio_np in generate_audio(
                    tts_provider, text, language, text_speaker
                ):
                    complete_audio = np.concatenate((complete_audio, audio_np))

                complete_audio = np.concatenate((complete_audio, np.zeros(2 * sr)))

                for sr, audio_np in generate_audio(
                    tts_provider, question, language, general_speaker
                ):
                    complete_audio = np.concatenate((complete_audio, audio_np))

                await context.store.set(AUDIO_DATA, (sr, complete_audio))

        case question_generator.LISTENING_COMPREHENSION:
            topic = await context.store.get(LISTENING_COMPREHENSION_TOPIC, None)
            speakers = await context.store.get(LISTENING_COMPREHENSION_SPEAKERS, None)
            text = await context.store.get(LISTENING_COMPREHENSION_TEXT, None)
            question = await context.store.get(LISTENING_COMPREHENSION_QUESTION, None)

            if topic is None:
                return "You have not provided a topic for the listening comprehension. Please ensure that you have provided all the necessary information."
            if speakers is None:
                return "You have not provided a speakers for the listening comprehension. Please ensure that you have provided all the necessary information."
            if text is None:
                return "You have not provided a text for the listening comprehension. Please ensure that you have provided all the necessary information."
            if question is None:
                return "You have not provided a question for the listening comprehension. Please ensure that you have provided all the necessary information."

            extra_parameters = await context.store.get("extra_parameters")

            mode_switch = extra_parameters["mode_switch"]

            if mode_switch:
                complete_audio = np.array([])
                tts_provider = extra_parameters["tts_provider"]
                language = extra_parameters["language"]

                first_speaker_voice_id = get_random_voice_id_for_provider(
                    tts_provider, language
                )
                second_speaker_voice_id = get_random_voice_id_for_provider(
                    tts_provider, language, exclude_ids=[first_speaker_voice_id]
                )

                third_speaker_voice_id = get_random_voice_id_for_provider(
                    tts_provider,
                    language,
                    exclude_ids=[first_speaker_voice_id, second_speaker_voice_id],
                )

                text_voices = [first_speaker_voice_id, second_speaker_voice_id]

                # Topic audio
                for sr, audio_np in generate_audio(
                    tts_provider, topic, language, third_speaker_voice_id
                ):
                    complete_audio = np.concatenate((complete_audio, audio_np))

                complete_audio = np.concatenate((complete_audio, np.zeros(2 * sr)))

                i = 0

                for text_segment in text:
                    audio_text = text_segment["text"]

                    for sr, audio_np in generate_audio(
                        tts_provider,
                        audio_text,
                        language,
                        voice_id=text_voices[i % 2],
                    ):
                        complete_audio = np.concatenate((complete_audio, audio_np))
                    i += 1
                    # Add silence between segments
                    complete_audio = np.concatenate((complete_audio, np.zeros(sr)))

                complete_audio = np.concatenate((complete_audio, np.zeros(2 * sr)))

                # Question audio
                for sr, audio_np in generate_audio(
                    tts_provider, question, language, third_speaker_voice_id
                ):
                    complete_audio = np.concatenate((complete_audio, audio_np))

                await context.store.set(AUDIO_DATA, (sr, complete_audio))

    return "The process was finished."


async def create_base_text(
    context: Context, potential_texts: list[list[str, str]]
) -> str:
    """This function is used to register the initial base text of the question.
    It takes a list of tuples which contain the possible options alongside the grammar structure the option focuses on.
    One of the options will then be selected
    The list should contain atleast 5 elements.

    The input values for potential_texts must never contain the placeholder in the from of "___".

    Args:
        context (Context): context
        text (list[list[str, str]]): A list of tuples which contain in the first spot the question text and the second spot the grammar structure this option focuses on. Should contain atleast 5 elements.

    Returns:
        str: The selected base text and further instructions.
    """

    if len(potential_texts) < 5:
        return "Please ensure that the potential_texts list contains atleast 5 options."

    if "___" in potential_texts[0]:
        return 'Your respons includes the placeholder marker "___". Please repeat this step and ensure not to include "___" in any option.'

    # Select one option
    option = potential_texts[random.randint(0, len(potential_texts) - 1)]

    await context.store.set(QUESTION_BASE_TEXT, option[0])

    question_type = await context.store.get("question_type")

    match question_type:
        case question_generator.TRANSLATION:
            return "You have done everything you can now finish up."
        case _:
            return _CREATE_QUESTION_BASE_TEXT_INSTRUCTION.format(
                text=option[0], focus=option[1]
            )


async def create_question_with_placholder(
    context: Context, question_text: str, answer: str
) -> str:
    """Registers a question.
    The input question_text of the create_question_with_placholder tool MUST contain the original sentence with a part being replaced by "___".
    The replaced part MUST be set as the answer.

    Grammatically interesting sections should be replaced rather than single nouns.

    Args:
        context (Context): context
        modified_text (str): The text of the question which should have a placeholder segment inside of it.
        answer (str): The replaced part.

    Returns:
        str: The next instruction
    """
    if not question_text.__contains__("___"):
        return 'Your question text was malformed and does not contain the placeholder for the answer which is: "___". Please correct this.'

    await context.store.set(QUESTION_TEXT, question_text)
    # await context.store.set(QUESTION_OPTIONS, options)
    await context.store.set(QUESTION_ANSWER, answer)

    return "You MUST now generate a hint for the answer. The hint MUST be in english."


async def create_question_hint(context: Context, hint: str) -> str:
    """This function is used to create hint for a give question.

    You should give a hint which helps in answering the question. For example when you replaced a verb it might give information about the expected form.

    The hint should be in english.

    The hint MUST only be written in keypoints and SHOULD NOT be a sentence.

    The hint MUST not contain newline characters. If you want to seperate multiple keypoints use commas.

    Args:
        context (Context): context
        hint (str): Hint to help solve the question.

    Returns:
        str: Next instruction
    """

    await context.store.set(QUESTION_HINT, hint)

    question_type = await context.store.get("question_type")

    match question_type:
        case question_generator.MULTI_CHOICE:
            return "You must now generate the incorrect options for the question type."

        case _:
            return "You have done everything you can now finish up."


async def create_multiple_choice_question_incorrect_options(
    context: Context, additional_options: list[str]
) -> str:
    """This function is used to register options for a multiple choice question.

    The options should be similar to the answer but they must not fit the criteria of the hint.
    None of these options you give should make sense if they are used as an answer for the question.

    The options must not include the correct answer.
    The options must exactly contain three values, not more and not less.

    Args:
        context (Context): context
        additional_options (list[str]): List of wrong options for the question.

    Returns:
        str: Next instruction
    """

    answer = await context.store.get(QUESTION_ANSWER)

    additional_options.insert(random.randint(0, len(additional_options)), answer)

    answer_index = additional_options.index(answer)

    await context.store.set(QUESTION_OPTIONS, additional_options)
    await context.store.set(QUESTION_ANSWER_INDEX, answer_index)

    return "You have done everything you can now finish up."


async def create_reading_comprehension_topic(
    context: Context, topics: list[str]
) -> str:
    """Creates a topic of a reading comprehension problem.

    It takes a list of potential topics from which one will be selected.
    The topics should be diverse and different from each other.

    Args:
        context (Context): context
        topics (list[str]): List of potential topics. Should contain atleast 5 elements.

    Returns:
        str: Tool response
    """
    if len(topics) < 5:
        return "Please ensure that the topics list contains atleast 5 options."

    topic = topics[random.randint(0, len(topics) - 1)]

    await context.store.set(READING_COMPREHENSION_TOPIC, topic)

    return _CREATE_COMPREHENSION_BASE_TEXT_INSTRUCTION.format(text=topic)


async def create_reading_comprehension_text(context: Context, text: str) -> str:
    """Create the text for the choosen topic.

    The generated text should be relevant to the topic.
    It MUST NOT add the topic at the beginning of the text.

    Args:
        context (Context): context
        text (str): The text for the given topic.

    Returns:
        str: Tool response
    """

    await context.store.set(READING_COMPREHENSION_TEXT, text)

    return "Now generate a question for the given text."


async def create_reading_comprehension_question(context: Context, question: str) -> str:
    """Create a question for give reading comprehension problem.

    Args:
        context (Context): context
        question (str): The question based on the topic and text.

    Returns:
        str: Tool response
    """

    await context.store.set(READING_COMPREHENSION_QUESTION, question)

    return "You have done everything you can now finish up."


async def create_listening_comprehension_topic(
    context: Context, topics: list[str]
) -> str:
    """Creates a topic of a comprehension problem.

    It takes a list of potential topics from which one will be selected.
    The topics should be diverse and different from each other.

    Args:
        context (Context): context
        topics (list[str]): List of potential topics. Should contain atleast 5 elements.

    Returns:
        str: Tool response
    """
    if len(topics) < 5:
        return "Please ensure that the topics list contains atleast 5 options."

    topic = topics[random.randint(0, len(topics) - 1)]

    await context.store.set(LISTENING_COMPREHENSION_TOPIC, topic)

    return _CREATE_COMPREHENSION_BASE_TEXT_INSTRUCTION.format(text=topic)


async def create_listening_comprehension_speakers(
    context: Context, speakers: list[str]
) -> str:
    """Creates a list of speakers for the comprehension problem.

    It takes a list of two speakers which will be the speakers of the comprehension problem.

    Args:
        context (Context): context
        speakers (list[str]): List of speakers. Should contain exactly 2 elements.

    Returns:
        str: Tool response
    """

    await context.store.set(LISTENING_COMPREHENSION_SPEAKERS, speakers)

    return f"In the next step you will generate the text for the comprehension problem. Please use the speakers you just provided. Speaker 1 is {speakers[0]} and speaker 2 is {speakers[1]}."


async def create_listening_comprehension_text(
    context: Context, text: list[dict]
) -> str:
    """Create the text for the choosen topic.

    The generated text should be relevant to the topic.
    It MUST NOT add the topic at the beginning of the text.

    The text should be a list of dictionaries with the following structure:
    [
        {
            "text": "The text",
            "speaker": "First speaker",
        },
        {
            "text": "The text",
            "speaker": "Second speaker",
        },
        ...
    ]

    Args:
        context (Context): context
        text (list[dict]): The text for the given topic.

    Returns:
        str: Tool response
    """

    await context.store.set(LISTENING_COMPREHENSION_TEXT, text)

    return "Now generate a question for the given text."


async def create_listening_comprehension_question(
    context: Context, question: str
) -> str:
    """Create a question for give reading comprehension problem.

    Args:
        context (Context): context
        question (str): The question based on the topic and text.

    Returns:
        str: Tool response
    """

    await context.store.set(LISTENING_COMPREHENSION_QUESTION, question)

    return "You have done everything you can now finish up."
