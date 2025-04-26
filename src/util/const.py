IS_STREAM: str = "is_stream"
AUDIO_OUTPUT: str = "audio_output"
MODEL: str = "model"
PROVIDER: str = "provider"
EMBEDDING_MODEL: str = "embedding"
SEARCH_ENGINE: str = "serach_engine"

OptionType = dict[str, tuple[str, str]]

PROVIDER_OPENROUTER = "openrouter"
PROVIDER_DEEPSEEK = "deekseek"
PROVIDER_OPENAI = "openai"

OPTION_MODEL: OptionType = {
    "OpenAI: GPT-4o-mini": ("gpt-4o-mini-2024-07-18", PROVIDER_OPENAI),
    "OpenAI: GPT-4.1 mini": ("gpt-4.1-mini-2025-04-14", PROVIDER_OPENAI),
    "OpenAI: GPT-4.1 nano": ("gpt-4.1-nano-2025-04-14", PROVIDER_OPENAI),
    "OpenAI: GPT-4.1": ("gpt-4.1-2025-04-14", PROVIDER_OPENAI),
    "Deepseek: DeepSeek-V3": ("deepseek-chat", PROVIDER_DEEPSEEK),
    "Deepseek: DeepSeek-R1": ("deepseek-reasoner", PROVIDER_DEEPSEEK),
}

OPTION_EMBEDDING: OptionType = [
    (
        "MiniLM L12 v2 (fast)",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ),
    ("BGE M3 (kind of slow)", "BAAI/bge-m3"),
]

NONE = "none"
GOOGLE = "google"
TAVILY = "tavily"
OPTION_SEARCH_ENGINE = [("Google", GOOGLE), ("Tavily", TAVILY), ("None", NONE)]


TTS_OPENAI = "TTS Openai"
TTS_KOKORO = "TTS Kokoro (local)"
TTS_ELEVENLABS = "TTS Elevenlabs"
TTS_FISH_AUDIO = "TTS Fish Audio (local)"
