"""
Утилитарные функции для работы с аудио.
"""

import os
import subprocess
import wave
import numpy as np
from scipy.signal import resample_poly
import logging
from typing import Tuple

logger = logging.getLogger('app.audio_utils')


def load_audio(file_path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Загрузка аудиофайла с использованием встроенной библиотеки wave.

    Args:
        file_path: Путь к аудиофайлу.
        sr: Целевая частота дискретизации.

    Returns:
        Кортеж (массив numpy, частота дискретизации).
    """
    try:
        with wave.open(file_path, 'rb') as wav_file:
            if wav_file.getnchannels() != 1:
                logger.warning("Файл %s не моно-аудио", file_path)

            frames = wav_file.readframes(-1)
            audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            sampling_rate = wav_file.getframerate()

            if sampling_rate != sr:
                gcd = np.gcd(sr, sampling_rate)
                audio_array = resample_poly(audio_array, sr // gcd, sampling_rate // gcd)
                sampling_rate = sr

            return audio_array, sampling_rate

    except Exception as e:
        logger.error("Ошибка при загрузке аудио %s: %s", file_path, e)
        raise


def get_audio_duration(file_path: str) -> float:
    """
    Определяет длительность аудиофайла с использованием ffprobe.

    Args:
        file_path: Путь к аудиофайлу.

    Returns:
        Длительность в секундах.
    """
    if not os.path.exists(file_path):
        raise Exception(f"Файл не существует: {file_path}")

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration:stream=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
        for line in result.stdout.strip().splitlines():
            stripped = line.strip()
            if stripped and stripped != "N/A":
                return float(stripped)
    except subprocess.TimeoutExpired:
        raise Exception(f"Таймаут при определении длительности файла {file_path}")
    except subprocess.CalledProcessError as e:
        raise Exception(f"Ошибка ffprobe для файла {file_path}: {e.stderr}")
    except (ValueError, TypeError) as e:
        raise Exception(f"Ошибка при преобразовании длительности для файла {file_path}: {e}")

    # ffprobe returned N/A — fall back to wave header for WAV files
    try:
        with wave.open(file_path, 'rb') as wav_file:
            return wav_file.getnframes() / wav_file.getframerate()
    except Exception:
        pass

    logger.warning("Не удалось определить длительность файла %s, возвращаем 0.0", file_path)
    return 0.0
