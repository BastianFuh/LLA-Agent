IS_STREAM: str = "is_stream"
AUDIO_OUTPUT: str = "audio_output"
MODEL: str = "model"
SEARCH_ENGINE: str = "serach_engine"

OPTION_MODEL: list[tuple[str, str]] = [
    ("DeepSeek: V3 0324 (free)", "deepseek/deepseek-chat-v3-0324:free"),
    ("DeepSeek: V3 (free)", "deepseek/deepseek-chat:free"),
    ("DeepSeek: R1 (free)", "deepseek/deepseek-r1:free"),
    ("Mistral: Mistral Nemo (free)", "mistralai/mistral-nemo:free"),
    ("Meta: Llama 3.1 8B Instruct (free)", "meta-llama/llama-3.1-8b-instruct:free"),
    (
        "Gemini Flash 2.0 Experimental (free) WARNING: MIGHT NOT WORK BECAUSE OF UPTIME PROBLEMS",
        "google/gemini-2.0-flash-exp:free",
    ),
    (
        "Gemini Flash 2.5 Experimental (free) WARNING: MIGHT NOT WORK BECAUSE OF UPTIME PROBLEMS",
        "google/gemini-2.5-pro-exp-03-25:free",
    ),
    (
        "Gemini Pro 2.0 Experimental (free) WARNING: MIGHT NOT WORK BECAUSE OF UPTIME PROBLEMS",
        "google/gemini-2.0-pro-exp-02-05:free",
    ),
]

NONE = "none"
GOOGLE = "google"
TAVILY = "tavily"
OPTION_SEARCH_ENGINE = [("Google", GOOGLE), ("Tavily", TAVILY), ("None", NONE)]
