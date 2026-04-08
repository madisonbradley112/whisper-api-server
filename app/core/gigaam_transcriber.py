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
        self.model_path = config.get("gigaam_variant", "v3_e2e_rnnt")
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
        Загрузка модели GigaAM.

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
                    result = self.model.transcribe_longform(audio_path)
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
                # transcribe() возвращает строку
                text = result if isinstance(result, str) else str(result)
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

        # Определяем длительность аудио
        duration = get_audio_duration(input_path)

        # Транскрибация (GigaAM сам загружает и обрабатывает аудио)
        result = self.transcribe(input_path, return_timestamps=return_timestamps,
                                 language=language, temperature=temperature,
                                 prompt=prompt, _duration=duration)

        elapsed_time = time.time() - start_time
        logger.info("Обработка и транскрибация завершены за %.2f секунд", elapsed_time)

        return result, duration
