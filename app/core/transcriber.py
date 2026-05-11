import time
import threading
import traceback
import zlib
from typing import Dict, Union
import logging

from faster_whisper import WhisperModel

from ..audio.processor import AudioProcessor
from ..infrastructure.storage import cleanup_temp_files

logger = logging.getLogger('app.transcriber')


class WhisperTranscriber:

    def __init__(self, config: Dict):
        self.config = config
        self.model_path = config["model_path"]
        self.language = config["language"]
        self.chunk_length_s = config["chunk_length_s"]
        self.batch_size = config["batch_size"]
        self.max_new_tokens = config["max_new_tokens"]
        self.return_timestamps = config["return_timestamps"]
        self.temperature = config["temperature"]

        # faster-whisper is not thread-safe; serialize inference calls
        self._inference_lock = threading.Lock()

        self.audio_processor = AudioProcessor(config)
        self._load_model()

    def _load_model(self) -> None:
        logger.info("Loading faster-whisper model: %s", self.model_path)
        # CTranslate2 does not support MPS; use CPU with int8 on Apple Silicon.
        # On Linux with CUDA the model will use float16 automatically.
        self.model = WhisperModel(
            self.model_path,
            device="cpu",
            compute_type="int8",
        )
        logger.info("Model loaded and ready")

    def _filter_hallucinations(self, text: str) -> str:
        if not text or not text.strip():
            return text
        encoded = text.encode("utf-8")
        ratio = len(encoded) / len(zlib.compress(encoded))
        if ratio < 2.4:
            logger.warning(
                "Rejected probable hallucination (compression_ratio=%.2f): %s",
                ratio, text[:80],
            )
            return ""
        return text

    def transcribe(self, audio_path: str, return_timestamps: bool = None,
                   language: str = None, temperature: float = None,
                   prompt: str = None) -> Union[str, Dict]:
        if return_timestamps is None:
            return_timestamps = self.return_timestamps
        if language is None:
            language = self.language
        if temperature is None:
            temperature = self.temperature

        logger.info("Transcribing: %s", audio_path)

        try:
            with self._inference_lock:
                segments_iter, info = self.model.transcribe(
                    audio_path,
                    language=language if language != "auto" else None,
                    task="transcribe",
                    beam_size=5,
                    temperature=temperature,
                    initial_prompt=prompt,
                    # Hallucination guards — no_speech_threshold set high so segments
                    # aren't silently dropped; compression_ratio catches repetition loops
                    vad_filter=False,
                    no_speech_threshold=0.9,
                    compression_ratio_threshold=2.4,
                    condition_on_previous_text=False,
                    word_timestamps=return_timestamps,
                )
                segments = list(segments_iter)

            logger.info("Audio duration seen by faster-whisper: %.2fs, segments: %d",
                        info.duration, len(segments))

            full_text = " ".join(s.text.strip() for s in segments)
            full_text = self._filter_hallucinations(full_text)

            if not return_timestamps:
                logger.info("Transcription done: %d chars", len(full_text))
                return full_text

            result_segments = []
            for seg in segments:
                seg_text = self._filter_hallucinations(seg.text.strip())
                if seg_text:
                    result_segments.append({
                        "start_time_ms": int(seg.start * 1000),
                        "end_time_ms": int(seg.end * 1000),
                        "text": seg_text,
                    })

            logger.info("Transcription done: %d segments", len(result_segments))
            return {"segments": result_segments, "text": full_text}

        except Exception as e:
            logger.error("Transcription error for '%s': %s", audio_path, e)
            logger.error("Traceback: %s", traceback.format_exc())
            raise

    def process_file(self, input_path: str, return_timestamps: bool = None,
                     language: str = None, temperature: float = None,
                     prompt: str = None) -> Union[str, Dict]:
        start = time.time()
        logger.info("Processing file: %s", input_path)
        temp_files = []
        try:
            processed_path, temp_files = self.audio_processor.process_audio(input_path)
            result = self.transcribe(processed_path, return_timestamps=return_timestamps,
                                     language=language, temperature=temperature,
                                     prompt=prompt)
            logger.info("Done in %.2fs", time.time() - start)
            return result
        except Exception as e:
            logger.error("Error processing '%s' after %.2fs: %s",
                         input_path, time.time() - start, e)
            logger.error("Traceback: %s", traceback.format_exc())
            raise
        finally:
            cleanup_temp_files(temp_files)
