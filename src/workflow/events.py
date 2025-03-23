from llama_index.core.workflow import Event


class LLM_Progress_Event(Event):
    response_delta: str
