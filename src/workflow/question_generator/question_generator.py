from llama_index.llms.openrouter import OpenRouter
from llama_index.core.tools import FunctionTool


from llama_index.core.agent.react import ReActChatFormatter
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context

from workflow.question_generator import tools as QGT

from pathlib import Path

import random
import sys


class QuestionGenerator:
    BASE_PROMPT = (
        (Path(__file__).parents[0] / Path("prompts") / Path("base.md"))
        .open("r", encoding="utf-8")
        .read()
    )

    MULTIPLE_CHOICE_PROMPT = (
        (Path(__file__).parents[0] / Path("prompts") / Path("multiple_choice.md"))
        .open("r", encoding="utf-8")
        .read()
    )

    def __init__(self, model: str):
        self.llm = OpenRouter(model=model)

    def get_agent(self, **kwargs) -> ReActAgent:
        return ReActAgent(
            tools=[
                FunctionTool.from_defaults(QGT.create_base_text),
                FunctionTool.from_defaults(QGT.create_multiplechoice_question),
            ],
            llm=self.llm,
            # chat_history=build_message(None, ev.history),
            formatter=ReActChatFormatter.from_defaults(
                system_header=self.BASE_PROMPT.format(
                    tool_desc="{tool_desc}",
                    tool_names="{tool_names}",
                    language=kwargs["language"],
                    language_proficiency=kwargs["language_proficiency"],
                    difficulty=kwargs["difficulty"],
                    additional_information=kwargs["additional_information"],
                )
                .replace("{'", "{{'")
                .replace('{"', '{{"')
                .replace("5}", "5}}")
                # These replace functions ensure that the example jsons will not be detected as a replacable token
                # in a later format call. Howerver this is still incredible jank and there might is a better option.
            ),
        )

    async def generate_multiple_choice(
        self,
        language: str,
        language_proficiency: str,
        difficulty: str,
        additional_information: str,
    ) -> dict:
        random_number = random.randint(0, sys.maxsize)

        additional_information = f"You should generate a random question, use this seed for the randomness  {random_number}. {additional_information}"

        agent = self.get_agent(
            language=language,
            language_proficiency=language_proficiency,
            difficulty=difficulty,
            additional_information=additional_information,
        )

        ctx = Context(agent)

        await agent.run(self.MULTIPLE_CHOICE_PROMPT, ctx=ctx)

        output = {
            QGT.QUESTION_BASE_TEXT: await ctx.get(QGT.QUESTION_BASE_TEXT),
            QGT.QUESTION_TEXT: await ctx.get(QGT.QUESTION_TEXT),
            QGT.QUESTION_OPTIONS: await ctx.get(QGT.QUESTION_OPTIONS),
            QGT.QUESTION_ANSWERS: await ctx.get(QGT.QUESTION_ANSWERS),
            QGT.QUESTION_HINT: await ctx.get(QGT.QUESTION_HINT),
        }

        return output
