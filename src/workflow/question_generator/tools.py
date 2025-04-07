from typing import Literal
from llama_index.core.workflow import Context
import random

QUESTION_BASE_TEXT = "question_base_text"
QUESTION_TEXT = "question_text"
QUESTION_OPTIONS = "question_options"
QUESTION_ANSWER = "question_answer"
QUESTION_ANSWER_INDEX = "question_answer_index"
QUESTION_HINT = "question_hint"


_CREATE_BASE_TEXT_INSTRUCTION = """
You have started the question generation with this text:

```
{text}
```

Now you must take the generated text and replace a word from it with the placeholder token: "___". You also must generate the correct answer.
"""


async def create_base_text(context: Context, potential_texts: list[str]) -> str:
    """This function is used to register the initial base text of the question.
    It takes a list of possible options and will select one of them.
    The list should contain atleast 5 elements.

    The input values for potential_texts must never contain the placeholder in the from of "___".

    Args:
        context (Context): context
        text (list[str]): A list of option which will form the base text. Should contain atleast 5 elements.

    Returns:
        str: The selected base text and further instructions.
    """

    if len(potential_texts) < 5:
        return "Please ensure that the potential_texts list contains atleast 5 options."

    # Select one option
    text = potential_texts[random.randint(3, len(potential_texts) - 1)]

    await context.set(QUESTION_BASE_TEXT, text)
    return _CREATE_BASE_TEXT_INSTRUCTION.format(text=text)


async def create_question_with_placholder(
    context: Context, question_text: str, answer: str
) -> str:
    """Registers a question.
    The input text of this tool should have a placeholder in the form of "___" which represents a part of an original sentence which has to be filled by the correct answer.

    The input consists of the modified text and the solutions

    Args:
        context (Context): context
        modified_text (str): The text of the question which should have a placeholder segment inside of it.
        answer (str): The answer to the modified text.

    Returns:
        str: The next instruction
    """
    if not question_text.__contains__("___"):
        return 'Your question text was malformed and does not contain the placeholder for the answer which is: "___". Please correct this.'

    await context.set(QUESTION_TEXT, question_text)
    # await context.set(QUESTION_OPTIONS, options)
    await context.set(QUESTION_ANSWER, answer)

    return "You must now generate a hint for the answer."


async def create_question_hint(
    context: Context, hint: str, question_type: Literal["multiple_choice", "free_text"]
) -> str:
    """This function is used to create hint for a give question.

    You should give a hint which helps in answering the question. For example when you replaced a verb it might give information about the expected form.

    The hint should be in english.

    The hint should only be written in keypoints and should not be a sentence.

    The hint must not contain newline characters. If you want to seperate multiple keypoints use commas.

    Args:
        context (Context): context
        hint (str): Hint to help solve the question.
        question_type (Literal): The question type. It is either multiple_choice or free_text

    Returns:
        str: Next instruction
    """

    await context.set(QUESTION_HINT, hint)

    if question_type == "multiple_choice":
        return "You must now generate the incorrect options for the question type."

    if question_type == "free_text":
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

    answer = await context.get(QUESTION_ANSWER)

    additional_options.insert(random.randint(0, len(additional_options) - 1), answer)

    answer_index = additional_options.index(answer)

    await context.set(QUESTION_OPTIONS, additional_options)
    await context.set(QUESTION_ANSWER_INDEX, answer_index)

    return "You have done everything you can now finish up."
