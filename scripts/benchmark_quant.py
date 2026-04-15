#!/usr/bin/env python3
"""
Бенчмарк квантизации Whisper-модели через CTranslate2 / faster-whisper.

Конвертирует исходную HuggingFace-модель в 4 формата (float16, int8_float16, int8, int4)
и прогоняет каждый вариант на всех аудиофайлах из указанной директории.
Выводит таблицу сравнения скорости, потребления VRAM и RTF,
а также транскрипции для визуальной оценки качества.

Зависимости (установить на сервере):
    pip install faster-whisper ctranslate2

Использование:
    python benchmark_quant.py \
        --model /home/text-generation/models/whisper/antony-ties-podlodka-v1.2 \
        --audio-dir /home/text-generation/bench/whisper \
        --output-dir /home/text-generation/models/whisper
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import torch

QUANTIZATIONS = ["float16", "int8_float16", "int8", "int4"]
AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".opus", ".aac"}


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _build_fake_flash_attn(base_dir: str) -> None:
    """Создаёт заглушку flash_attn пакета в base_dir для подпроцесса конвертации.

    Настоящий файловый Python-пакет надёжнее sys.meta_path мока: не вмешивается
    во внутренний import-механизм torch._library и inspect.
    """
    pkg = os.path.join(base_dir, "flash_attn")
    os.makedirs(pkg)

    # Корневой __init__.py — содержит атрибуты, которые проверяет transformers
    with open(os.path.join(pkg, "__init__.py"), "w") as f:
        f.write(
            '__version__ = "2.6.0"\n'
            "flash_attn_func = None\n"
            "flash_attn_varlen_func = None\n"
            "flash_attn_with_kvcache = None\n"
        )

    # Субмодули, которые импортирует transformers
    submodules = {
        "bert_padding": "index_first_axis = None\npad_input = None\nunpad_input = None\n",
        "flash_attn_interface": "flash_attn_func = None\nflash_attn_varlen_func = None\n",
    }
    for name, content in submodules.items():
        subdir = os.path.join(pkg, name)
        os.makedirs(subdir)
        with open(os.path.join(subdir, "__init__.py"), "w") as f:
            f.write(content)

    # flash_attn.layers — отдельный субпакет с rotary и другими модулями
    layers_dir = os.path.join(pkg, "layers")
    os.makedirs(layers_dir)
    with open(os.path.join(layers_dir, "__init__.py"), "w") as f:
        f.write("RotaryEmbedding = None\n")
    with open(os.path.join(layers_dir, "rotary.py"), "w") as f:
        f.write("RotaryEmbedding = None\napply_rotary_emb = None\n")

    # flash_attn_2_cuda как отдельный модуль (не субпакет flash_attn)
    with open(os.path.join(base_dir, "flash_attn_2_cuda.py"), "w") as f:
        f.write("# stub\n")


# ---------------------------------------------------------------------------
# Конвертация
# ---------------------------------------------------------------------------

def convert_model(src_path: str, out_dir: str, quantization: str) -> str:
    """Конвертирует HF-модель в CTranslate2 формат с заданной квантизацией.

    Запускает конвертацию в подпроцессе с отключённым flash_attn, чтобы избежать
    конфликта ABI между flash_attn_2_cuda и ctranslate2.

    Возвращает путь к сконвертированной модели.
    Пропускает конвертацию, если директория уже существует и не пуста.
    """
    model_name = Path(src_path).name
    dest = os.path.join(out_dir, f"{model_name}-ct2-{quantization}")

    if os.path.isdir(dest) and any(Path(dest).iterdir()):
        print(f"  [skip] уже сконвертирована: {dest}")
        return dest

    print(f"  Конвертация в {quantization} → {dest}")
    t = time.time()

    # Конвертация в подпроцессе с поддельным flash_attn пакетом.
    # Почему подпроцесс: ctranslate2 импортирует transformers, который безусловно тянет
    # flash_attn_2_cuda.so, скомпилированный под другую версию libtorch.
    # Решение: создаём настоящий Python-пакет flash_attn на диске и ставим его первым
    # в PYTHONPATH — Python находит его раньше сломанного пакета в conda.
    tmpdir = tempfile.mkdtemp(prefix="fake_flash_attn_")
    try:
        _build_fake_flash_attn(tmpdir)
        env = os.environ.copy()
        env["PYTHONPATH"] = tmpdir + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

        # Monkey-patch: ctranslate2 4.7.1 передаёт dtype= вместо torch_dtype= в from_pretrained,
        # что несовместимо с transformers 4.49.0 — WhisperForConditionalGeneration.__init__
        # не принимает dtype как аргумент конструктора.
        script = """
import ctranslate2

_orig = ctranslate2.converters.TransformersConverter.load_model
def _patched(self, model_class, model_name_or_path, **kwargs):
    if 'dtype' in kwargs:
        kwargs['torch_dtype'] = kwargs.pop('dtype')
    return _orig(self, model_class, model_name_or_path, **kwargs)
ctranslate2.converters.TransformersConverter.load_model = _patched

c = ctranslate2.converters.TransformersConverter({src!r}, low_cpu_mem_usage=True)
c.convert({dest!r}, quantization={quant!r})
print("ok")
""".format(src=src_path, dest=dest, quant=quantization)
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, env=env,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if result.returncode != 0:
        raise RuntimeError(result.stderr[-3000:])

    print(f"  Готово за {time.time() - t:.0f}с")
    return dest


# ---------------------------------------------------------------------------
# Бенчмарк
# ---------------------------------------------------------------------------

def get_audio_files(audio_dir: str) -> list[str]:
    """Возвращает отсортированный список аудиофайлов из директории."""
    return sorted(
        str(p)
        for p in Path(audio_dir).iterdir()
        if p.suffix.lower() in AUDIO_EXTENSIONS
    )


def run_benchmark(ct2_path: str, audio_files: list[str], language: str, runs: int) -> dict:
    """Загружает модель и прогоняет её на всех аудиофайлах.

    Первый прогон — warm-up (не учитывается в статистике).
    Возвращает словарь {имя_файла: метрики}.
    """
    from faster_whisper import WhisperModel

    print(f"  Загрузка модели из {ct2_path} ...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t_load = time.time()
    model = WhisperModel(ct2_path, device="cuda", compute_type="auto", local_files_only=True)
    load_time = time.time() - t_load
    load_vram_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"  Загружена за {load_time:.1f}s | VRAM: {load_vram_gb:.2f} GB")

    file_results = {}

    for audio_path in audio_files:
        name = Path(audio_path).name
        times = []
        transcript = ""
        audio_duration = 0.0
        peak_vram_gb = 0.0

        for run_i in range(runs + 1):  # 0 = warm-up
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()

            segs_gen, info = model.transcribe(
                audio_path,
                language=language,
                beam_size=1,
                vad_filter=False,
                temperature=0.0,
            )
            text_parts = [s.text for s in segs_gen]  # материализуем генератор
            elapsed = time.time() - t0
            peak_vram_gb = torch.cuda.max_memory_allocated() / 1024 ** 3

            if run_i == 0:
                # warm-up: сохраняем транскрипт, время не учитываем
                transcript = " ".join(text_parts).strip()
                audio_duration = info.duration
                print(f"    warm-up {name}: {elapsed:.2f}s")
            else:
                times.append(elapsed)
                print(f"    run {run_i}/{runs}  {name}: {elapsed:.2f}s  peak VRAM: {peak_vram_gb:.2f} GB")

        avg = sum(times) / len(times) if times else 0.0
        file_results[name] = {
            "avg_s": avg,
            "min_s": min(times) if times else 0.0,
            "peak_vram_gb": peak_vram_gb,
            "rtf": avg / audio_duration if audio_duration else 0.0,
            "audio_duration_s": audio_duration,
            "transcript": transcript,
        }

    del model
    torch.cuda.empty_cache()
    return file_results


# ---------------------------------------------------------------------------
# Вывод таблицы
# ---------------------------------------------------------------------------

def print_results(all_results: dict[str, dict]) -> None:
    """Выводит сводную таблицу и транскрипции."""
    quants = list(all_results.keys())
    file_names = list(next(iter(all_results.values())).keys())

    W = 112
    SEP = "-" * W

    print("\n" + "=" * W)
    print("  РЕЗУЛЬТАТЫ БЕНЧМАРКА")
    print("=" * W)

    for fname in file_names:
        baseline = all_results.get("float16", {}).get(fname, {})
        baseline_avg = baseline.get("avg_s") or 0.0
        audio_dur = baseline.get("audio_duration_s", 0.0)

        print(f"\n  Файл: {fname}  (длительность: {audio_dur:.1f}s)")
        print(SEP)
        print(f"  {'Квантизация':<22} {'avg (s)':<10} {'min (s)':<10} {'RTF':<8} {'peak VRAM (GB)':<16} {'Ускорение':>10}")
        print(SEP)

        for q in quants:
            r = all_results[q].get(fname, {})
            avg = r.get("avg_s", 0.0)
            min_t = r.get("min_s", 0.0)
            rtf = r.get("rtf", 0.0)
            vram = r.get("peak_vram_gb", 0.0)
            if q == "float16" or baseline_avg == 0:
                speedup_str = "baseline"
            else:
                speedup_str = f"{baseline_avg / avg:.2f}x" if avg else "—"
            print(f"  {q:<22} {avg:<10.2f} {min_t:<10.2f} {rtf:<8.3f} {vram:<16.2f} {speedup_str:>10}")

        print(SEP)

    # Транскрипции
    print("\n" + "=" * W)
    print("  ТРАНСКРИПЦИИ (warm-up прогон, первые 250 символов)")
    print("=" * W)

    for fname in file_names:
        print(f"\n  Файл: {fname}")
        for q in quants:
            t = all_results[q].get(fname, {}).get("transcript", "")
            preview = t[:250] + ("…" if len(t) > 250 else "")
            print(f"  [{q:<16}]  {preview}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Бенчмарк квантизации Whisper через CTranslate2 / faster-whisper"
    )
    parser.add_argument("--model", required=True,
                        help="Путь к исходной HuggingFace-модели")
    parser.add_argument("--audio-dir", required=True,
                        help="Директория с тестовыми аудиофайлами")
    parser.add_argument("--output-dir", required=True,
                        help="Директория для сохранения конвертированных моделей")
    parser.add_argument("--language", default="ru",
                        help="Язык транскрибации (default: ru)")
    parser.add_argument("--runs", type=int, default=3,
                        help="Кол-во измерительных прогонов на файл, не считая warm-up (default: 3)")
    parser.add_argument("--quantizations", nargs="+", default=QUANTIZATIONS,
                        choices=QUANTIZATIONS,
                        help="Какие уровни квантизации тестировать (default: все)")
    parser.add_argument("--skip-convert", action="store_true",
                        help="Пропустить конвертацию — использовать уже существующие модели")
    args = parser.parse_args()

    # Аудиофайлы
    audio_files = get_audio_files(args.audio_dir)
    if not audio_files:
        print(f"Нет аудиофайлов в {args.audio_dir}")
        return
    print(f"\nНайдено аудиофайлов: {len(audio_files)}")
    for f in audio_files:
        print(f"  {f}")

    model_name = Path(args.model).name
    ct2_paths: dict[str, str] = {}

    # Конвертация
    print("\n--- КОНВЕРТАЦИЯ ---")
    for q in args.quantizations:
        dest = os.path.join(args.output_dir, f"{model_name}-ct2-{q}")
        if args.skip_convert:
            ct2_paths[q] = dest
        else:
            try:
                ct2_paths[q] = convert_model(args.model, args.output_dir, q)
            except Exception as e:
                print(f"  ОШИБКА конвертации {q}: {e}")
                traceback.print_exc()

    # Бенчмарк
    all_results: dict[str, dict] = {}
    print("\n--- БЕНЧМАРК ---")
    for q in args.quantizations:
        if q not in ct2_paths:
            print(f"\n[{q}] пропущен (ошибка конвертации)")
            continue
        print(f"\n[{q}]")
        try:
            all_results[q] = run_benchmark(ct2_paths[q], audio_files, args.language, args.runs)
        except Exception as e:
            print(f"  ОШИБКА бенчмарка {q}: {e}")
            traceback.print_exc()

    if all_results:
        print_results(all_results)
    else:
        print("\nНет результатов для вывода.")


if __name__ == "__main__":
    main()
