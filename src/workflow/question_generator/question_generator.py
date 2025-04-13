from llama_index.core.tools import FunctionTool


from llama_index.core.agent.react import ReActChatFormatter
from llama_index.core.agent.workflow import ReActAgent, FunctionAgent
from llama_index.core.workflow import Context
from llama_index.core.llms.function_calling import FunctionCallingLLM

from workflow.question_generator import tools as QGT

from workflow import utils

import prompts


class QuestionGenerator:
    def __init__(self, model: str):
        self.llm = utils.get_llm(model)

    def _get_agent(self, **kwargs) -> ReActAgent:
        tools = [
            FunctionTool.from_defaults(QGT.create_base_text),
            FunctionTool.from_defaults(QGT.create_question_with_placholder),
            FunctionTool.from_defaults(QGT.create_question_hint),
            FunctionTool.from_defaults(
                QGT.create_multiple_choice_question_incorrect_options
            ),
        ]

        if isinstance(self.llm, FunctionCallingLLM):
            prompt = (
                prompts.QUESTION_GENERATOR_BASE_FUNCTION_PROMPT.format(
                    language=kwargs["language"],
                    language_proficiency=kwargs["language_proficiency"],
                    difficulty=kwargs["difficulty"],
                    additional_information=kwargs["additional_information"],
                )
                # These replace functions ensure that the example jsons will not be detected as a replacable token
                # in a later format call. Howerver this is still incredible jank and there might is a better option.
                .replace("{'", "{{'")
                .replace('{"', '{{"')
                .replace("5}", "5}}")
            )

            return FunctionAgent(
                name="Chatbot Agent",
                description="Todo",
                tools=tools,
                llm=self.llm,
                system_prompt=prompt,
            )
        else:
            prompt = (
                prompts.QUESTION_GENERATOR_BASE_REACT_PROMPT.format(
                    tool_desc="{tool_desc}",
                    tool_names="{tool_names}",
                    language=kwargs["language"],
                    language_proficiency=kwargs["language_proficiency"],
                    difficulty=kwargs["difficulty"],
                    additional_information=kwargs["additional_information"],
                )
                # These replace functions ensure that the example jsons will not be detected as a replacable token
                # in a later format call. Howerver this is still incredible jank and there might is a better option.
                .replace("{'", "{{'")
                .replace('{"', '{{"')
                .replace("5}", "5}}")
            )

            return ReActAgent(
                tools=tools,
                llm=self.llm,
                formatter=ReActChatFormatter.from_defaults(system_header=prompt),
            )

    async def _run_agent(
        self,
        question_type: str,
        language: str,
        language_proficiency: str,
        difficulty: str,
        additional_information: str,
    ) -> Context:
        agent = self._get_agent(
            language=language,
            language_proficiency=language_proficiency,
            difficulty=difficulty,
            additional_information=additional_information,
        )

        ctx = Context(agent)
        await ctx.set("question_type", question_type)

        await agent.run(
            prompts.QUESTION_GENERATOR_INITIAL_MESSAGE_PROMPT.format(
                question_type=question_type
            ),
            ctx=ctx,
        )

        return ctx

    async def generate_multiple_choice(
        self,
        language: str,
        language_proficiency: str,
        difficulty: str,
        additional_information: str,
    ) -> dict:
        ctx = await self._run_agent(
            "multiple_choice",
            language,
            language_proficiency,
            difficulty,
            additional_information,
        )

        output = {
            QGT.QUESTION_BASE_TEXT: await ctx.get(QGT.QUESTION_BASE_TEXT),
            QGT.QUESTION_TEXT: await ctx.get(QGT.QUESTION_TEXT),
            QGT.QUESTION_OPTIONS: await ctx.get(QGT.QUESTION_OPTIONS),
            QGT.QUESTION_ANSWER_INDEX: await ctx.get(QGT.QUESTION_ANSWER_INDEX),
            QGT.QUESTION_HINT: await ctx.get(QGT.QUESTION_HINT),
        }

        return output

    async def generate_free_text(
        self,
        language: str,
        language_proficiency: str,
        difficulty: str,
        additional_information: str,
    ) -> dict:
        ctx = await self._run_agent(
            "free_text",
            language,
            language_proficiency,
            difficulty,
            additional_information,
        )

        output = {
            QGT.QUESTION_BASE_TEXT: await ctx.get(QGT.QUESTION_BASE_TEXT),
            QGT.QUESTION_TEXT: await ctx.get(QGT.QUESTION_TEXT),
            QGT.QUESTION_ANSWER: await ctx.get(QGT.QUESTION_ANSWER),
            QGT.QUESTION_HINT: await ctx.get(QGT.QUESTION_HINT),
        }

        return output

    async def generate_translation_question(
        self,
        language: str,
        language_proficiency: str,
        difficulty: str,
        additional_information: str,
    ) -> dict:
        ctx = await self._run_agent(
            "translation",
            language,
            language_proficiency,
            difficulty,
            additional_information,
        )

        output = {
            QGT.QUESTION_BASE_TEXT: await ctx.get(QGT.QUESTION_BASE_TEXT),
        }

        return output
