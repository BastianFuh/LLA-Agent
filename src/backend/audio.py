import json
import logging
import random
import re
from pathlib import Path
from typing import Generator, Tuple

import numpy as np
from elevenlabs import ElevenLabs
from faster_whisper import WhisperModel
from kokoro import KPipeline
from matplotlib.backend_bases import NonGuiException
from openai import OpenAI
from pydub import AudioSegment

from util import const, get_language_code

logger = logging.getLogger(__name__)

voice_path = (
    Path(__file__).parents[0] / Path("..") / Path("..") / Path("voice_references")
)

OPENAI_VOICES = [
    "alloy",
    "ballad",
    "coral",
    "echo",
    "fable",
    "onyx",
    "nova",
    "sage",
    "shimmer",
    "verse",
]

KOKORO_VOICES = ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"]


def _convert_mp3_to_wav(mp3_path: Path):
    """
    Convert mp3 to wav using pydub
    """

    wav_path = mp3_path.with_suffix(".wav")
    if not wav_path.exists():
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format="wav")
    return wav_path


def _convert_all_mp3_to_wav():
    """
    Convert all mp3 files in the voice_path to wav
    """
    for file in voice_path.glob("*.mp3"):
        _convert_mp3_to_wav(file)


def _transcribe_all_files():
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    for file in voice_path.glob("*.wav"):
        # Only transcribe if the json does not exist
        if (file.with_suffix(".json")).exists():
            logger.info(f"Skipping {file}")
            continue

        segments, info = model.transcribe(str(file), beam_size=5)

        text = ""
        for segment in segments:
            text += segment.text

        meta_data = {
            "text": text,
            "language": info.language,
        }

        try:
            with open(file.with_suffix(".json"), "w", encoding="utf-8") as f:
                json.dump(meta_data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.info(f"Error writing to file {file.with_suffix('.json')}: {e}")
            # delete the file if it exists
            if (file.with_suffix(".json")).exists():
                (file.with_suffix(".json")).unlink()
                logger.info(f"Deleted {file.with_suffix('.json')}")


def init_fish_audio_voice_samples():
    """
    Convert all mp3 files in the voice_path to wav and transcribe them
    """
    _convert_all_mp3_to_wav()
    _transcribe_all_files()


def get_fish_audio_voice_samples(language: str) -> list[dict[Path, str]]:
    """
    Get the voice samples for a specific language
    """
    language_code = get_language_code(language)
    if not language_code:
        logger.info(f"Found no voices for language {language}")
        return []

    # Get the files from voice_path and iterate over them
    voice_samples = []
    for file in voice_path.glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if data["language"] == language_code:
                voice_samples.append(
                    {
                        "audio_file": file.with_suffix(".wav"),
                        "text": data["text"],
                    }
                )

    return voice_samples


def _get_id_with_exluded_ids(num_voices: int, exclude_ids: list[int] = []) -> int:
    voice_ids = list(range(num_voices))

    for voice_id in exclude_ids:
        if voice_id in voice_ids:
            voice_ids.remove(voice_id)

    if len(voice_ids) == 0:
        return 0

    return random.choice(voice_ids)


def get_random_voice_id_for_provider(
    tts_provider: str, language: str, exclude_ids: list[int] = []
) -> int | None:
    """
    Get the voice samples for a specific language and TTS provider
    """

    if tts_provider == const.TTS_KOKORO:
        return _get_id_with_exluded_ids(len(KOKORO_VOICES), exclude_ids)

    if tts_provider == const.TTS_OPENAI:
        return _get_id_with_exluded_ids(len(OPENAI_VOICES), exclude_ids)

    if tts_provider == const.TTS_FISH_AUDIO:
        return _get_id_with_exluded_ids(
            len(get_fish_audio_voice_samples(language)), exclude_ids
        )

    return NonGuiException


def _generate_audio_kokoro(
    text: str, language: str, voice_id: int = -1
) -> Generator[Tuple[int, np.ndarray], None, None]:
    pipeline = KPipeline(lang_code="j", repo_id="hexgrad/Kokoro-82M")

    if voice_id < 0:
        voice_id = get_random_voice_id_for_provider(const.TTS_KOKORO, language)

    generator = pipeline(
        text,
        voice=KOKORO_VOICES[voice_id],
        speed=1,
        # Split into sections so that the audio has some small pauses on punctuation
        split_pattern=r"(\.|。|!|\?|！|？|、)",
    )

    for gs, ps, audio in generator:
        yield (24000, audio.numpy())


def _generate_audio_elevenlabs(
    text: str,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    from io import BytesIO

    client = ElevenLabs()

    audio = client.text_to_speech.convert(
        text=text,
        voice_id="GxxMAMfQkDlnqjpzjLHH",
        model_id="eleven_flash_v2_5",
        output_format="pcm_24000",
    )

    stream = BytesIO()

    for chunk in audio:
        if chunk:
            stream.write(chunk)
    stream.seek(0)

    # return_audio = np.frombuffer(audio, dtype=np.int16)
    yield (24000, np.frombuffer(stream.getbuffer(), dtype=np.int16))


def _generate_audio_openai(
    text: str, language: str, voice_id: int = -1
) -> Generator[Tuple[int, np.ndarray], None, None]:
    instructions = f"""Language: Speak in {language} with a standard accent.

                Voice Affect: Composed; project authority and confidence.

                Tone: Sincere and authoritative—express genuine apology while conveying competence.

                Pacing: Steady and moderate.

                Pronunciation: Clear and precise."""

    if voice_id < 0:
        voice_id = random.randint(0, len(OPENAI_VOICES) - 1)

    client = OpenAI()
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=OPENAI_VOICES[voice_id],
        input=text,
        instructions=instructions,
        response_format="pcm",
    )
    return_audio = np.frombuffer(response.content, dtype=np.int16)

    yield (24000, return_audio)


def _generate_audio_fish_audio(
    text: str, language: str, voice_id: int = -1
) -> Generator[Tuple[int, np.ndarray], None, None]:
    import io
    import wave

    from fish_audio_sdk import ReferenceAudio, Session, TTSRequest

    client = Session(apikey="LOCAL", base_url="http://127.0.0.1:8080")

    wave_file = io.BytesIO()

    voices = get_fish_audio_voice_samples(language)

    if voice_id < 0:
        voice_id = random.randint(0, len(voices) - 1)

    if len(voices) != 0:
        # Select a random voice from the list
        voice = voices[voice_id]

        # Prepare the voice reference

        audio_path = voice["audio_file"]

        # Read wav file to bytes
        with open(audio_path, "rb") as f:
            voice_audio = f.read()

        references = [
            ReferenceAudio(
                audio=voice_audio,
                text=voice["text"],
            )
        ]
    else:
        references = []

    for chunk in client.tts(TTSRequest(text=text, format="wav", references=references)):
        wave_file.write(chunk)
    wave_file.seek(0)

    with wave.open(wave_file, "rb") as wav_file:
        num_channels = wav_file.getnchannels()
        framerate = wav_file.getframerate()
        num_frames = wav_file.getnframes()

        audio_data = wav_file.readframes(num_frames)

        audio_np = np.frombuffer(audio_data, dtype=np.int16)

        # If stereo (2 channels), reshape to [num_frames, 2] array
        if num_channels == 2:
            audio_np = audio_np.reshape(-1, 2)

    yield (framerate, audio_np)


def generate_audio(
    tts_provider: str, text: str, language: str, voice_id: int = -1
) -> Generator[Tuple[int, np.ndarray], None, None]:
    if tts_provider == const.TTS_KOKORO:
        for sr, audio_np in _generate_audio_kokoro(text, language, voice_id=voice_id):
            yield (sr, audio_np)

    if tts_provider == const.TTS_ELEVENLABS:
        for sr, audio_np in _generate_audio_elevenlabs(text):
            yield (sr, audio_np)

    if tts_provider == const.TTS_OPENAI:
        for sr, audio_np in _generate_audio_openai(text, language, voice_id=voice_id):
            yield (sr, audio_np)

    if tts_provider == const.TTS_FISH_AUDIO:
        for sr, audio_np in _generate_audio_fish_audio(
            text, language, voice_id=voice_id
        ):
            yield (sr, audio_np)
