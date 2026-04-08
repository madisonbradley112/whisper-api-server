"""
Модуль config.py содержит функции для управления конфигурацией приложения.
"""

import json
import os
import logging
from typing import Dict

logger = logging.getLogger('app.config')


def load_config(config_path: str) -> Dict:
    """
    Загружает конфигурацию из JSON-файла.
    
    Args:
        config_path: Путь к файлу конфигурации.
        
    Returns:
        Словарь с параметрами конфигурации.
        
    Raises:
        FileNotFoundError: Если файл конфигурации не найден.
        json.JSONDecodeError: Если файл конфигурации содержит некорректный JSON.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Валидация типа модели
        model_type = config.get("model_type", "whisper")
        if model_type not in ("whisper", "gigaam"):
            raise ValueError(f"Неизвестный model_type в конфигурации: {model_type}. "
                             f"Допустимые значения: whisper, gigaam")

        # Мержим секцию активной модели в плоский конфиг —
        # все модули продолжают читать config["model_path"], config["language"] и т.д.
        models = config.pop("models", {})
        model_config = models.get(model_type, {})
        config.update(model_config)

        logger.info("Конфигурация успешно загружена из %s (model_type=%s)", config_path, model_type)
        return config
    except FileNotFoundError as e:
        logger.error("Файл конфигурации не найден: %s", e)
        raise
    except json.JSONDecodeError as e:
        logger.error("Ошибка при загрузке конфигурации: %s", e)
        raise