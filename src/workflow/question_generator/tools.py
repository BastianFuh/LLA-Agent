from llama_index.core.workflow import Context
import random

QUESTION_BASE_TEXT = "question_base_text"
QUESTION_TEXT = "question_text"
QUESTION_OPTIONS = "question_options"
QUESTION_ANSWERS = "question_answers"
QUESTION_HINT = "question_hint"


_CREATE_BASE_TEXT_INSTRUCTION = """
You have started the question generation with this text:

```
{text}
```

Now you must take the generated text and replace a word from it with the placeholder token: "___". You also must generate a list of potential options for the question. It has to contain the correct answer and three incorrect answers. 
"""


async def create_base_text(context: Context, potential_texts: list[str]) -> str:
    """This function is used to register the initial base text of the question.
    It takes a list of possible options and will select one of them.
    The list should contain atleast 5 elements.

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


async def create_multiplechoice_question(
    context: Context,
    question_text: str,
    options: list[str],
    correct_answer: int,
    hint: str,
) -> str:
    """Registers a multiple choise question.
    The input text should have a placeholder in the form of "___" which represents a part of an
    original sentence which has to be filled by the correct answer.

    The answer will be given in a list alongside three incorrect options.

    Additionally the index of the correct answer inside this list is also given.

    You also should give a hint which helps in answering the question. For example when
    you replaced a verb it might give information about the expected form.
    The hint should be in english.
    The hint should only be written in keypoints and should not be a sentence.
    Make sure that the hint

    Args:
        context (Context): context
        modified_text (str): The text of the question which should have a placeholder segment inside of it.
        options (list[str]): A list which contains one correct answer and three incorrect answers.
        correct_answer (int): The list index of the correct answer.
        hint (str): A hint which helps in answering the answer.

    Returns:
        str: The success of operation.
    """
    if not question_text.__contains__("___"):
        return 'Your question text was malformed and does not contain the placeholder for the answer which is: "___". Please correct this.'

    if not len(options) == 4:
        return "The options list should only contain four options please correct this."

    if correct_answer > 3:
        return "The index for the correct answer is outside of the bound of the list. It should be between 0 and 3 and should be the index for the correct answer. Please correct this."

    await context.set(QUESTION_TEXT, question_text)
    await context.set(QUESTION_OPTIONS, options)
    await context.set(QUESTION_ANSWERS, correct_answer)
    await context.set(QUESTION_HINT, hint)

    return "You have successfully registered the question. You can now finish."
