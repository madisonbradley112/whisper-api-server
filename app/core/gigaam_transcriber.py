"""
Модуль gigaam_transcriber.py содержит класс GigaAMTranscriber для транскрибации
аудиофайлов с помощью модели GigaAM-v3 (Conformer-based ASR).
GigaAM самостоятельно обрабатывает аудио через ffmpeg, поэтому AudioProcessor
(sox-нормализация, добавление тишины) не используется.
"""

import time
import threading
import traceback
from typing import Dict, Tuple, Union
import logging

from ..audio.utils import get_audio_duration

logger = logging.getLogger('app.gigaam_transcriber')

# Порог длительности аудио (секунды), выше которого используется transcribe_longform
_LONGFORM_THRESHOLD_S = 25


class GigaAMTranscriber:
    """
    Класс для распознавания речи с помощью модели GigaAM.

    Attributes:
        config (Dict): Словарь с параметрами конфигурации.
        model_path (str): Идентификатор модели для gigaam.load_model().
        return_timestamps (bool): Флаг возврата временных меток по умолчанию.
    """

    def __init__(self, config: Dict):
        """
        Инициализация транскрайбера GigaAM.

        Args:
            config: Словарь с параметрами конфигурации.
        """
        self.config = config
        self.model_path = config.get("variant", "v3_e2e_rnnt")
        self.return_timestamps = config.get("return_timestamps", False)

        # Lock для потокобезопасного доступа к модели —
        # Waitress обслуживает запросы в нескольких потоках
        self._inference_lock = threading.Lock()

        # Флаг для однократного предупреждения об игнорируемых параметрах
        self._warned_ignored_params = False

        # Загружаем модель при инициализации
        self._load_model()

    def _load_model(self) -> None:
        """
        Загрузка модели GigaAM и модели сегментации для longform.

        Raises:
            Exception: Если не удалось загрузить модель.
        """
        import gigaam

        logger.info("Загрузка модели GigaAM: %s", self.model_path)

        try:
            self.model = gigaam.load_model(self.model_path)
            logger.info("Модель GigaAM успешно загружена и готова к использованию")
        except Exception as e:
            logger.error("Ошибка при загрузке модели GigaAM: %s", e)
            raise

        # Инициализация VAD-пайплайна для longform из явного пути
        segmentation_path = self.config.get("segmentation_model")
        if segmentation_path:
            self._init_vad_pipeline(segmentation_path)

    def _init_vad_pipeline(self, model_path: str) -> None:
        """
        Инициализация pyannote VAD-пайплайна из локального пути к модели.
        Также патчит segment_audio_file, чтобы передавать аудио как waveform —
        это обходит зависимость pyannote 4.x от torchcodec для декодирования файлов.
        """
        import gigaam.vad_utils as vad_utils
        from gigaam.preprocess import load_audio as gigaam_load_audio, SAMPLE_RATE
        from pyannote.audio import Model
        from pyannote.audio.pipelines import VoiceActivityDetection
        import torch

        logger.info("Загрузка модели сегментации из: %s", model_path)
        try:
            model = Model.from_pretrained(model_path)
            pipeline = VoiceActivityDetection(segmentation=model)
            pipeline.instantiate({"min_duration_on": 0.0, "min_duration_off": 0.0})
            vad_utils._PIPELINE = pipeline
            logger.info("Модель сегментации успешно загружена")
        except Exception as e:
            logger.error("Ошибка при загрузке модели сегментации: %s", e)
            raise

        # Патчим segment_audio_file: pyannote 4.x требует torchcodec для чтения файлов,
        # но torchcodec не работает с conda-окружением из-за старого GCC runtime.
        # Вместо этого загружаем аудио через ffmpeg (GigaAM preprocess) и передаём
        # в pipeline как waveform dict — pyannote принимает такой формат напрямую.
        def _patched_segment_audio_file(wav_file, sr, device=torch.device("cpu"), **kwargs):
            audio = gigaam_load_audio(wav_file)
            vad_pipeline = vad_utils.get_pipeline(device)
            waveform_dict = {"waveform": audio.unsqueeze(0), "sample_rate": SAMPLE_RATE}
            sad_segments = vad_pipeline(waveform_dict)

            # Остальная логика чанкинга — из оригинальной функции
            max_duration = kwargs.get("max_duration", 22.0)
            min_duration = kwargs.get("min_duration", 15.0)
            strict_limit_duration = kwargs.get("strict_limit_duration", 30.0)
            new_chunk_threshold = kwargs.get("new_chunk_threshold", 0.2)

            segments = []
            boundaries = []
            curr_duration = 0.0
            curr_start = 0.0
            curr_end = 0.0

            def _update_segments(cs, ce, cd):
                if cd > strict_limit_duration:
                    max_segs = int(cd / strict_limit_duration) + 1
                    seg_dur = cd / max_segs
                    ce_local = cs + seg_dur
                    for _ in range(max_segs - 1):
                        segments.append(audio[int(cs * sr): int(ce_local * sr)])
                        boundaries.append((cs, ce_local))
                        cs = ce_local
                        ce_local += seg_dur
                segments.append(audio[int(cs * sr): int(ce * sr)])
                boundaries.append((cs, ce))

            for segment in sad_segments.get_timeline().support():
                start = max(0, segment.start)
                end = min(audio.shape[0] / sr, segment.end)
                if curr_duration == 0.0:
                    curr_start = start
                elif curr_duration > new_chunk_threshold and (
                    curr_duration + (end - curr_end) > max_duration
                    or curr_duration > min_duration
                ):
                    _update_segments(curr_start, curr_end, curr_duration)
                    curr_start = start
                curr_end = end
                curr_duration = curr_end - curr_start

            if curr_duration > new_chunk_threshold:
                _update_segments(curr_start, curr_end, curr_duration)

            return segments, boundaries

        vad_utils.segment_audio_file = _patched_segment_audio_file
        logger.info("segment_audio_file пропатчен для обхода torchcodec")

    def _warn_ignored_params(self, language: str = None, temperature: float = None,
                             prompt: str = None) -> None:
        """Однократное предупреждение об игнорируемых параметрах."""
        if self._warned_ignored_params:
            return
        ignored = []
        if language:
            ignored.append(f"language={language}")
        if temperature is not None:
            ignored.append(f"temperature={temperature}")
        if prompt:
            ignored.append("prompt")
        if ignored:
            logger.debug("GigaAM не поддерживает параметры: %s — они будут проигнорированы",
                         ", ".join(ignored))
            self._warned_ignored_params = True

    def transcribe(self, audio_path: str, return_timestamps: bool = None,
                   language: str = None, temperature: float = None,
                   prompt: str = None, _duration: float = None) -> Union[str, Dict]:
        """
        Транскрибация аудиофайла.

        Args:
            audio_path: Путь к аудиофайлу.
            return_timestamps: Флаг возврата временных меток. Если None — берётся из конфига.
            language: Не используется (для совместимости интерфейса).
            temperature: Не используется (для совместимости интерфейса).
            prompt: Не используется (для совместимости интерфейса).
            _duration: Предвычисленная длительность (внутренний параметр, чтобы не считать дважды).

        Returns:
            В зависимости от параметра return_timestamps:
            - Если return_timestamps=False: строка с распознанным текстом
            - Если return_timestamps=True: словарь с ключами "segments" и "text"
        """
        if return_timestamps is None:
            return_timestamps = self.return_timestamps

        self._warn_ignored_params(language, temperature, prompt)

        logger.info("Начало транскрибации файла: %s", audio_path)

        try:
            # Определяем длительность для выбора метода транскрибации
            duration = _duration if _duration is not None else get_audio_duration(audio_path)
            use_longform = duration > _LONGFORM_THRESHOLD_S

            with self._inference_lock:
                if use_longform:
                    logger.info("Аудио %.1f с — используем transcribe_longform", duration)
                    result = self.model.transcribe_longform(audio_path, word_timestamps=return_timestamps)
                else:
                    if return_timestamps:
                        result = self.model.transcribe(audio_path, word_timestamps=True)
                    else:
                        result = self.model.transcribe(audio_path)

            # Форматируем результат
            if use_longform:
                return self._format_longform_result(result, return_timestamps)
            elif return_timestamps:
                return self._format_timestamps_result(result)
            else:
                # transcribe() возвращает TranscriptionResult с атрибутом .text
                text = result.text if hasattr(result, 'text') else str(result)
                logger.info("Транскрибация завершена: получено %s символов текста", len(text))
                return text

        except Exception as e:
            logger.error("Ошибка в процессе транскрибации аудиофайла '%s': %s", audio_path, e)
            logger.error("Тип исключения: %s", type(e).__name__)
            logger.error("Traceback: %s", traceback.format_exc())
            raise

    def _format_longform_result(self, segments_list, return_timestamps: bool) -> Union[str, Dict]:
        """
        Форматирование результата transcribe_longform().

        Args:
            segments_list: Список сегментов от GigaAM (каждый имеет .start, .end, .text).
            return_timestamps: Нужны ли временные метки в ответе.

        Returns:
            Строка или словарь с сегментами и текстом.
        """
        full_text = " ".join(seg.text for seg in segments_list)

        if not return_timestamps:
            logger.info("Транскрибация (longform) завершена: получено %s символов текста", len(full_text))
            return full_text

        segments = []
        for seg in segments_list:
            segments.append({
                "start_time_ms": int(seg.start * 1000),
                "end_time_ms": int(seg.end * 1000),
                "text": seg.text,
            })

        logger.info("Транскрибация (longform) с временными метками завершена: %s сегментов", len(segments))
        return {"segments": segments, "text": full_text}

    def _format_timestamps_result(self, result) -> Dict:
        """
        Форматирование результата transcribe(word_timestamps=True).

        Args:
            result: Объект с атрибутом .words (каждый word: .start, .end, .text).

        Returns:
            Словарь с ключами "segments" и "text".
        """
        segments = []
        words = result.words if hasattr(result, 'words') else []

        for word in words:
            segments.append({
                "start_time_ms": int(word.start * 1000),
                "end_time_ms": int(word.end * 1000),
                "text": word.text,
            })

        full_text = " ".join(w.text for w in words)
        logger.info("Транскрибация с временными метками завершена: %s сегментов", len(segments))
        return {"segments": segments, "text": full_text}

    def _extract_duration(self, result) -> float:
        """
        Извлечение длительности аудио из результата транскрибации.
        Используется как fallback, когда ffprobe не смог определить длительность.
        """
        try:
            # Результат longform — dict с сегментами или LongformTranscriptionResult
            if isinstance(result, dict) and result.get("segments"):
                last_seg = result["segments"][-1]
                return last_seg["end_time_ms"] / 1000.0
            # Результат longform до форматирования (итерируемый объект с .segments)
            if hasattr(result, 'segments') and result.segments:
                return result.segments[-1].end
        except Exception as e:
            logger.debug("Не удалось извлечь длительность из результата: %s", e)
        return 0.0

    def process_file(self, input_path: str, return_timestamps: bool = None,
                     language: str = None, temperature: float = None,
                     prompt: str = None) -> Tuple[Union[str, Dict], float]:
        """
        Полный процесс обработки и транскрибации аудиофайла.
        GigaAM самостоятельно обрабатывает аудио — AudioProcessor не используется.

        Args:
            input_path: Путь к исходному аудиофайлу.
            return_timestamps: Флаг возврата временных меток. Если None — берётся из конфига.
            language: Не используется (для совместимости интерфейса).
            temperature: Не используется (для совместимости интерфейса).
            prompt: Не используется (для совместимости интерфейса).

        Returns:
            Кортеж (результат транскрибации, длительность аудио в секундах).
        """
        start_time = time.time()
        logger.info("Начало обработки файла: %s", input_path)

        # Определяем длительность аудио (нефатально — при ошибке используем longform)
        duration_known = True
        try:
            duration = get_audio_duration(input_path)
        except Exception as e:
            logger.warning("Не удалось определить длительность: %s. Используем transcribe_longform", e)
            duration = _LONGFORM_THRESHOLD_S + 1
            duration_known = False

        # Транскрибация (GigaAM сам загружает и обрабатывает аудио)
        result = self.transcribe(input_path, return_timestamps=return_timestamps,
                                 language=language, temperature=temperature,
                                 prompt=prompt, _duration=duration)

        # Если длительность не была определена — пытаемся извлечь из результата
        if not duration_known:
            duration = self._extract_duration(result)

        elapsed_time = time.time() - start_time
        logger.info("Обработка и транскрибация завершены за %.2f секунд", elapsed_time)

        return result, duration
