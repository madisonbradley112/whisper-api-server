"""Модуль core — основные компоненты сервиса распознавания речи."""

from typing import Dict


def create_transcriber(config: Dict):
    """
    Создание транскрайбера на основе типа модели в конфигурации.

    Args:
        config: Словарь с параметрами конфигурации.

    Returns:
        Экземпляр WhisperTranscriber или GigaAMTranscriber.

    Raises:
        ValueError: Если указан неизвестный тип модели.
    """
    model_type = config.get("model_type", "whisper")

    if model_type == "whisper":
        from .transcriber import WhisperTranscriber
        return WhisperTranscriber(config)
    elif model_type == "gigaam":
        from .gigaam_transcriber import GigaAMTranscriber
        return GigaAMTranscriber(config)
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")
