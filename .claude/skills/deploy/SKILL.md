---
name: deploy
description: >
  Use when: user says "deploy", "деплой", "задеплой", "обнови сервер",
  "обнови на orange", "залей на сервер", "перезапусти whisper",
  "рестарт whisper", "статус whisper на orange", "логи whisper на orange".
  Do NOT use for: local development, testing, or unrelated server tasks.
---

# deploy: Деплой Whisper API на сервер Orange

## Инфраструктура

**Хост:** `orange` (10.10.1.20) -- bare metal, NVIDIA RTX 3090
**Путь на сервере:** `/home/text-generation/servers/whisper-api`
**Conda env:** `whisper-api` (Python 3.12)
**Systemd unit:** `whisper.service`
**Порт:** 5042
**Startup:** `server.sh` (activates conda, runs `python server.py --config config.json`)

## SSH-доступ

```bash
ssh orange         # обычный доступ: читать файлы, смотреть логи
ssh root@orange    # для systemctl restart/stop/start
```

## Поведенческие правила

- **Сравнивать перед копированием** -- всегда проверить какие файлы изменились между локальной версией и сервером
- **Не копировать config.json** -- конфиг на сервере может отличаться от локального (пути к модели, порт, device)
- **Не копировать history/, logs/, plans/** -- серверные данные не перезаписываем
- **После деплоя** -- всегда проверить `systemctl status whisper.service` и убедиться что модель загружена
- **requirements.txt** -- если изменился, установить зависимости перед рестартом

## Процедуры

### 1. Полный деплой (обновление файлов + рестарт)

```bash
# 1. Определить изменённые файлы (сравнить с main или указанным диапазоном коммитов)
git diff --name-only main..HEAD
# или если деплоим конкретный коммит:
git show --name-only --format="" <commit>

# 2. Скопировать изменённые файлы (кроме config.json, history/, logs/, plans/)
scp <file> orange:/home/text-generation/servers/whisper-api/<file>

# 3. Если изменился requirements.txt -- установить зависимости
ssh orange "source ~/.miniconda/etc/profile.d/conda.sh && conda activate whisper-api && \
  pip install -r /home/text-generation/servers/whisper-api/requirements.txt"

# 4. Перезапустить сервис
ssh root@orange "systemctl restart whisper.service"

# 5. Проверить статус (подождать ~15 сек на загрузку модели)
sleep 15
ssh root@orange "systemctl status whisper.service --no-pager"
```

### 2. Только рестарт (без обновления файлов)

```bash
ssh root@orange "systemctl restart whisper.service"
sleep 15
ssh root@orange "systemctl status whisper.service --no-pager"
```

### 3. Проверка статуса

```bash
ssh root@orange "systemctl status whisper.service --no-pager"
```

### 4. Логи

```bash
ssh root@orange "journalctl -u whisper.service -n 50 --no-pager"
# Реалтайм:
ssh root@orange "journalctl -u whisper.service -f"
```

### 5. Проверка API

```bash
# Health check
curl -s http://orange.lan:5042/v1/models

# Тест транскрипции (если есть тестовый файл)
curl -s -X POST http://orange.lan:5042/v1/audio/transcriptions \
  -F "file=@test.wav" -F "model=whisper"
```

### 6. Откат

```bash
# На сервере есть git -- можно откатить к конкретному коммиту
ssh orange "cd /home/text-generation/servers/whisper-api && git log --oneline -5"
ssh orange "cd /home/text-generation/servers/whisper-api && git checkout <commit> -- <file>"
ssh root@orange "systemctl restart whisper.service"
```

## Файлы, которые НЕ деплоим

| Файл/директория | Причина |
|-----------------|---------|
| `config.json` | Серверный конфиг отличается (пути, device, порт) |
| `history/` | Серверная история транскрипций |
| `logs/` | Серверные логи |
| `plans/` | Локальные планы разработки |
| `CLAUDE.md`, `RULES.md` | Только для разработки |
| `docs/`, `examples/` | Документация, не нужна на сервере |
| `client.png`, `README.md` | Не влияют на работу сервиса |
