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
**Рантайм:** Docker + NVIDIA Container Toolkit
**Compose-файл:** `docker-compose.yml` в корне проекта
**Порт:** 5042
**Код:** монтируется как volume (`.:/app`) — rebuild не нужен при изменениях кода

## SSH-доступ

```bash
ssh orange         # обычный доступ: читать файлы, смотреть логи, docker команды
ssh root@orange    # если нужны права root
```

## Поведенческие правила

- **Деплой = git pull на сервере** -- не копируем файлы через scp, сервер сам тянет из git
- **Не трогать config.json на сервере** -- серверный конфиг может отличаться от локального
- **После деплоя** -- проверить `docker compose ps` и убедиться что контейнер running
- **requirements.txt изменился** -- нужен `--build`, иначе достаточно `restart`
- **Модель грузится ~15-30 сек** после старта контейнера, подождать перед проверкой API

## Процедуры

### 1. Полный деплой (обновление кода + рестарт)

```bash
# Код обновляется через git pull на сервере -- rebuild не нужен
ssh orange "cd /home/text-generation/servers/whisper-api && git pull && docker compose restart"

# Проверить статус (подождать загрузку модели)
sleep 20
ssh orange "cd /home/text-generation/servers/whisper-api && docker compose ps"
```

### 2. Деплой с обновлением зависимостей (requirements.txt изменился)

```bash
ssh orange "cd /home/text-generation/servers/whisper-api && git pull && docker compose up -d --build"

sleep 30
ssh orange "cd /home/text-generation/servers/whisper-api && docker compose ps"
```

### 3. Только рестарт (без обновления кода)

```bash
ssh orange "cd /home/text-generation/servers/whisper-api && docker compose restart"
sleep 20
ssh orange "cd /home/text-generation/servers/whisper-api && docker compose ps"
```

### 4. Проверка статуса

```bash
ssh orange "cd /home/text-generation/servers/whisper-api && docker compose ps"
```

### 5. Логи

```bash
# Последние 50 строк
ssh orange "cd /home/text-generation/servers/whisper-api && docker compose logs --tail=50"

# Реалтайм
ssh orange "cd /home/text-generation/servers/whisper-api && docker compose logs -f"
```

### 6. Проверка API

```bash
# Health check
curl -s http://orange.lan:5042/v1/models

# Тест транскрипции
curl -s -X POST http://orange.lan:5042/v1/audio/transcriptions \
  -F "file=@test.wav" -F "model=whisper"
```

### 7. Откат

```bash
# Откатить код к конкретному коммиту и перезапустить
ssh orange "cd /home/text-generation/servers/whisper-api && git checkout <commit> && docker compose restart"
```

## Первоначальная настройка (один раз)

Если Docker ещё не развёрнут на сервере:

```bash
# Убедиться что NVIDIA Container Toolkit установлен
ssh orange "docker info | grep -i runtime"

# Остановить старый systemd сервис
ssh root@orange "systemctl disable --now whisper.service"

# Первый билд (долгий: ~10-15 мин)
ssh orange "cd /home/text-generation/servers/whisper-api && docker compose up -d --build"
```
