from typing import AsyncGenerator, Callable

from llama_index.core.tools import FunctionTool

from llama_index.core.agent.react import ReActChatFormatter
from llama_index.core.agent.workflow import ReActAgent, FunctionAgent
from llama_index.core.workflow import Context
from llama_index.core.llms.function_calling import FunctionCallingLLM

import queue

from hashlib import md5

from workflow.question_generator import tools as QGT
from workflow import question_generator

from workflow import utils

import prompts


class QuestionBuffer:
    DEFAULT_SETTINGS = [""]

    def __init__(self):
        self.buffer: dict[str, queue.Queue] = dict()

    def _create_storage(self) -> queue.Queue:
        return queue.Queue()

    def _hash_settings(self, settings: list[str]) -> str:
        return md5("".join(settings).encode(), usedforsecurity=False).hexdigest()

    def _create_entry(self, value, settings: list[str]) -> str:
        settings_hash = self._hash_settings(settings)

        return (value, settings_hash)

    def _validate_settings(self, entry, settings: list[str]) -> bool:
        settings_hash = self._hash_settings(settings)
        return entry[1] == settings_hash

    def add(self, key, question, settings: list[str] = DEFAULT_SETTINGS):
        if not self.buffer.keys().__contains__(key):
            self.buffer[key] = self._create_storage()

        self.buffer[key].put(self._create_entry(question, settings))

    def empty(self, key):
        if not self.buffer.keys().__contains__(key):
            return True

        return self.buffer[key].qsize() == 0

    def get(self, key, settings: list[str] = DEFAULT_SETTINGS):
        try:
            entry = self.buffer[key].get_nowait()
        except queue.Empty:
            return None
        else:
            self.buffer[key].task_done()

        if self._validate_settings(entry, settings):
            return entry[0]
        else:
            return None


class QuestionGenerator:
    def __init__(self, model: str, buffer: QuestionBuffer = QuestionBuffer()):
        self.llm = utils.get_llm(model)
        self.question_buffer = buffer

    def _get_initial_message_prompt(self, question_type) -> str:
        match question_type:
            case question_generator.MULTI_CHOICE:
                return prompts.INITIAL_MESSAGE_MULTIPLE_CHOICE

            case question_generator.FREE_TEXT:
                return prompts.INITIAL_MESSAGE_SIMPLE_FREE_TEXT

            case question_generator.TRANSLATION:
                return prompts.INITIAL_MESSAGE_TRANSLATION

            case _:
                raise ValueError("Unknown question type")

    def _get_agent(self, **kwargs) -> ReActAgent:
        tools = [
            FunctionTool.from_defaults(QGT.finish, return_direct=True),
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

        initial_message_prompt = self._get_initial_message_prompt(question_type)

        await agent.run(
            initial_message_prompt,
            ctx=ctx,
        )

        return ctx

    def generate_multiple_choice(
        self,
        language: str,
        language_proficiency: str,
        difficulty: str,
        additional_information: str,
    ) -> AsyncGenerator[dict, None]:
        return self._generate_question(
            question_generator.MULTI_CHOICE,
            language,
            language_proficiency,
            difficulty,
            additional_information,
        )

    def generate_free_text(
        self,
        language: str,
        language_proficiency: str,
        difficulty: str,
        additional_information: str,
    ) -> AsyncGenerator[dict, None]:
        return self._generate_question(
            question_generator.FREE_TEXT,
            language,
            language_proficiency,
            difficulty,
            additional_information,
        )

    def generate_translation_question(
        self,
        language: str,
        language_proficiency: str,
        difficulty: str,
        additional_information: str,
    ) -> AsyncGenerator[dict, None]:
        return self._generate_question(
            question_generator.TRANSLATION,
            language,
            language_proficiency,
            difficulty,
            additional_information,
        )

    async def _handle_agent_result(self, ctx: Context, key: str) -> dict:
        match key:
            case question_generator.MULTI_CHOICE:
                return {
                    QGT.QUESTION_BASE_TEXT: await ctx.get(QGT.QUESTION_BASE_TEXT),
                    QGT.QUESTION_TEXT: await ctx.get(QGT.QUESTION_TEXT),
                    QGT.QUESTION_OPTIONS: await ctx.get(QGT.QUESTION_OPTIONS),
                    QGT.QUESTION_ANSWER_INDEX: await ctx.get(QGT.QUESTION_ANSWER_INDEX),
                    QGT.QUESTION_HINT: await ctx.get(QGT.QUESTION_HINT),
                }

            case question_generator.FREE_TEXT:
                return {
                    QGT.QUESTION_BASE_TEXT: await ctx.get(QGT.QUESTION_BASE_TEXT),
                    QGT.QUESTION_TEXT: await ctx.get(QGT.QUESTION_TEXT),
                    QGT.QUESTION_ANSWER: await ctx.get(QGT.QUESTION_ANSWER),
                    QGT.QUESTION_HINT: await ctx.get(QGT.QUESTION_HINT),
                }

            case question_generator.TRANSLATION:
                return {
                    QGT.QUESTION_BASE_TEXT: await ctx.get(QGT.QUESTION_BASE_TEXT),
                }

            case _:
                raise ValueError(f"{key}: Unknown key value")

    async def _generate_question(
        self,
        key: str,
        language: str,
        language_proficiency: str,
        difficulty: str,
        additional_information: str,
    ) -> AsyncGenerator[dict, None]:
        settings = [language, language_proficiency, difficulty, additional_information]
        generation_count = 2
        yielded_result = False

        if not self.question_buffer.empty(key):
            buffer_value = self.question_buffer.get(
                key,
                settings,
            )

            if buffer_value is not None:
                yield buffer_value
                generation_count = 1
                yielded_result = True

        for i in range(generation_count):
            ctx = await self._run_agent(
                key,
                language,
                language_proficiency,
                difficulty,
                additional_information,
            )

            output = await self._handle_agent_result(ctx, key)

            if not yielded_result:
                yield output
                yielded_result = True
            else:
                self.question_buffer.add(key, output, settings)
