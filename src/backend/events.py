import numpy as np
from llama_index.core.workflow import Event, StartEvent


class ChatBotStartEvent(StartEvent):
    message: str
    history: dict | list | None


class LLMStartEvent(Event):
    message: str
    history: dict | list | None


class LLMFinishedEvent(Event):
    result: str | None


class LLMProgressEvent(Event):
    response: str


class AudioStartEvent(Event):
    text: str


class AudioStreamEvent(Event):
    audio_chunk: tuple[int, np.array]


class AudioFinishedEvent(Event):
    pass
