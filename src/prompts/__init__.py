from pathlib import Path

QUESTION_GENERATOR_BASE_REACT_PROMPT = (
    (Path(__file__).parents[0] / Path("question_generator") / Path("react_base.md"))
    .open("r", encoding="utf-8")
    .read()
)

QUESTION_GENERATOR_BASE_FUNCTION_PROMPT = (
    (
        Path(__file__).parents[0]
        / Path("question_generator")
        / Path("function_calling_base.md")
    )
    .open("r", encoding="utf-8")
    .read()
)

QUESTION_GENERATOR_INITIAL_MESSAGE_PROMPT = (
    (
        Path(__file__).parents[0]
        / Path("question_generator")
        / Path("initial_message.md")
    )
    .open("r", encoding="utf-8")
    .read()
)

QUESTION_GENERATOR_TRANSLATION_REACT_CHATBOT_PROMPT = (
    (
        Path(__file__).parents[0]
        / Path("question_generator")
        / Path("translation")
        / Path("react_base_translation.md")
    )
    .open("r", encoding="utf-8")
    .read()
)

QUESTION_GENERATOR_TRANSLATION_FUNCTION_CHATBOT_PROMPT = (
    (
        Path(__file__).parents[0]
        / Path("question_generator")
        / Path("translation")
        / Path("function_base_translation.md")
    )
    .open("r", encoding="utf-8")
    .read()
)

QUESTION_GENERATOR_TRANSLATION_REQUEST_PROMPT = (
    (
        Path(__file__).parents[0]
        / Path("question_generator")
        / Path("translation")
        / Path("input_translation.md")
    )
    .open("r", encoding="utf-8")
    .read()
)

CHATBOT_REACT_PROMPT = (
    (Path(__file__).parents[0] / Path("chatbot") / Path("react_prompt.md"))
    .open("r", encoding="utf-8")
    .read()
)

CHATBOT_FUNCTION_PROMPT = (
    (Path(__file__).parents[0] / Path("chatbot") / Path("function_prompt.md"))
    .open("r", encoding="utf-8")
    .read()
)

INITIAL_MESSAGE_MULTIPLE_CHOICE = (
    (
        Path(__file__).parents[0]
        / Path("question_generator")
        / Path("multiple_choice")
        / Path("initial_message.md")
    )
    .open("r", encoding="utf-8")
    .read()
)

INITIAL_MESSAGE_SIMPLE_FREE_TEXT = (
    (
        Path(__file__).parents[0]
        / Path("question_generator")
        / Path("simple_free_text")
        / Path("initial_message.md")
    )
    .open("r", encoding="utf-8")
    .read()
)

INITIAL_MESSAGE_TRANSLATION = (
    (
        Path(__file__).parents[0]
        / Path("question_generator")
        / Path("translation")
        / Path("initial_message.md")
    )
    .open("r", encoding="utf-8")
    .read()
)
