import logging
import threading
from typing import Any, Callable, Tuple

import numpy as np
from RealtimeSTT import AudioToTextRecorder

logger = logging.getLogger(__name__)


class AudioTranscriber:
    end_of_sentence_detection_pause = 0.45
    unknown_sentence_detection_pause = 0.7
    mid_sentence_detection_pause = 2.0

    _recorder_config = {
        "use_microphone": False,
        "spinner": False,
        "model": "large-v3",  # or large-v2 or deepdml/faster-whisper-large-v3-turbo-ct2 or ...
        "download_root": None,  # default download root location. Ex. ~/.cache/huggingface/hub/ in Linux
        "realtime_model_type": "tiny",  # or small.en or distil-small.en or ...
        "language": "",
        "silero_sensitivity": 0.05,
        "webrtc_sensitivity": 3,
        "post_speech_silence_duration": unknown_sentence_detection_pause,
        "min_length_of_recording": 1.1,
        "min_gap_between_recordings": 0,
        "enable_realtime_transcription": True,
        "realtime_processing_pause": 0.02,
        "silero_deactivity_detection": True,
        "early_transcription_on_silence": 0,
        "beam_size": 5,
        "beam_size_realtime": 3,
        "no_log_file": True,
        "initial_prompt_realtime": (
            "End incomplete sentences with ellipses.\n"
            "Examples:\n"
            "Complete: The sky is blue.\n"
            "Incomplete: When the sky...\n"
            "Complete: She walked home.\n"
            "Incomplete: Because he...\n"
        ),
        "silero_use_onnx": True,
        "faster_whisper_vad_filter": False,
    }

    def __init__(
        self, handle_output_while_not_recording: Callable[[Any], str | dict] = None
    ) -> None:
        self.recorder = self._create_recorder()

        self.full_sentences = []
        self.displayed_text = ""
        self.prev_text = ""

        self.reseting_state = False

        self.recording = False
        self.not_started = True

        self.handle_output_while_not_recording = handle_output_while_not_recording

        threading.Thread(target=self._handle_audio, daemon=True).start()

    def _handle_audio(self) -> None:
        while True:
            self.recorder.text(self._process_text)

    def _create_recorder(self) -> AudioToTextRecorder:
        _recorder_config = self._recorder_config.copy()
        _recorder_config["on_realtime_transcription_update"] = self._text_detected
        return AudioToTextRecorder(**_recorder_config)

    def _preprocess_text(self, text: str) -> str:
        # Remove leading whitespaces
        text = text.lstrip()

        #  Remove starting ellipses if present
        if text.startswith("..."):
            text = text[3:]

        # Remove any leading whitespaces again after ellipses removal
        text = text.lstrip()

        # Uppercase the first letter
        if text:
            text = text[0].upper() + text[1:]

        return text

    def _text_detected(self, text: str) -> None:
        text = self._preprocess_text(text)

        sentence_end_marks = [".", "!", "?", "。"]
        if text.endswith("..."):
            self.recorder.post_speech_silence_duration = (
                self.mid_sentence_detection_pause
            )
        elif (
            text
            and text[-1] in sentence_end_marks
            and self.prev_text
            and self.prev_text[-1] in sentence_end_marks
        ):
            self.recorder.post_speech_silence_duration = (
                self.end_of_sentence_detection_pause
            )
        else:
            self.recorder.post_speech_silence_duration = (
                self.unknown_sentence_detection_pause
            )

        self.prev_text = text

        new_displayed_text = " ".join(self.full_sentences)

        # If the current text is not a sentence-ending, display it in real-time
        if text:
            new_displayed_text += text

        if new_displayed_text != self.displayed_text:
            self.displayed_text = new_displayed_text

    def _process_text(self, text: str) -> None:
        self.recorder.post_speech_silence_duration = (
            self.unknown_sentence_detection_pause
        )

        text = self._preprocess_text(text)
        text = text.rstrip()
        if text.endswith("..."):
            text = text[:-2]

        if not text:
            return

        self.full_sentences.append(text)
        self.prev_text = ""
        self._text_detected("")

    def start_recording(self) -> None:
        logger.info("Starting recording...")
        self.recording = True
        self.recorder.start()

    def stop_recording(self) -> None:
        self.recording = False
        self.full_sentences = []
        self.displayed_text = ""
        self.prev_text = ""
        self.recorder.clear_audio_queue()
        self.recorder.stop()
        logger.info("Stopped recording...")

    def feed_audio(self, audio_chunk: Tuple[int, np.ndarray]) -> None:
        self.recorder.feed_audio(audio_chunk[1], original_sample_rate=audio_chunk[0])

    def get_text(self, *args) -> str:
        if self.recording:
            return self.displayed_text
        else:
            if self.handle_output_while_not_recording:
                return self.handle_output_while_not_recording(*args)
            else:
                return ""
