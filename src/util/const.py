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
    "DeepSeek: V3 0324 (free)": (
        "deepseek/deepseek-chat-v3-0324:free",
        PROVIDER_OPENROUTER,
    ),
    "DeepSeek: V3 (free)": ("deepseek/deepseek-chat:free", PROVIDER_OPENROUTER),
    "DeepSeek: R1 (free)": ("deepseek/deepseek-r1:free", PROVIDER_OPENROUTER),
    "DeepSeek: V3 0324": ("deepseek/deepseek-chat-v3-0324", PROVIDER_OPENROUTER),
    "DeepSeek: V3": ("deepseek/deepseek-chat", PROVIDER_OPENROUTER),
    "DeepSeek: R1": ("deepseek/deepseek-r1", PROVIDER_OPENROUTER),
    "Mistral: Mistral Nemo (free)": (
        "mistralai/mistral-nemo:free",
        PROVIDER_OPENROUTER,
    ),
    "Meta: Llama 3.1 8B Instruct (free)": (
        "meta-llama/llama-3.1-8b-instruct:free",
        PROVIDER_OPENROUTER,
    ),
    "Gemini Flash 2.0 Experimental (free) WARNING: MIGHT NOT WORK BECAUSE OF UPTIME PROBLEMS": (
        "google/gemini-2.0-flash-exp:free",
        PROVIDER_OPENROUTER,
    ),
    "Gemini Flash 2.5 Experimental (free) WARNING: MIGHT NOT WORK BECAUSE OF UPTIME PROBLEMS": (
        "google/gemini-2.5-pro-exp-03-25:free",
        PROVIDER_OPENROUTER,
    ),
    "Gemini Pro 2.0 Experimental (free) WARNING: MIGHT NOT WORK BECAUSE OF UPTIME PROBLEMS": (
        "google/gemini-2.0-pro-exp-02-05:free",
        PROVIDER_OPENROUTER,
    ),
    "OpenAI: GPT-4o-mini": ("gpt-4o-mini-2024-07-18", PROVIDER_OPENAI),
    "OpenAI: GPT-4.1 mini": ("gpt-4.1-mini-2025-04-14", PROVIDER_OPENAI),
    "OpenAI: GPT-4.1 nano": ("gpt-4.1-nano-2025-04-14", PROVIDER_OPENAI),
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
