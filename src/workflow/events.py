from llama_index.core.workflow import Event, StartEvent, StopEvent


class LLM_StartEvent(StartEvent):
    message: str
    history: dict | list | None


class LLMStopEvent(StopEvent):
    result: str


class LLM_Progress_Event(Event):
    response_delta: str
