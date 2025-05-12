import json
import logging
from pathlib import Path

from faster_whisper import WhisperModel
from pydub import AudioSegment

from util import get_language_code

logger = logging.getLogger(__name__)

voice_path = (
    Path(__file__).parents[0] / Path("..") / Path("..") / Path("voice_references")
)


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


# get voice samples for a specific language
def get_voice_samples(language: str) -> list[dict[Path, str]]:
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
