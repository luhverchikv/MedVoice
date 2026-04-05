# Vosk Medical Dataset Tester

Инструмент для тестирования моделей распознавания речи Vosk на медицинских фразах.

## Структура проекта

```
vosk_medical_test/
├── database.py         # Управление SQLite базой данных
├── tts_generator.py   # Генерация аудио из текста
├── vosk_recognizer.py # Распознавание речи с Vosk
├── main.py             # Главный скрипт (CLI интерфейс)
├── requirements.txt    # Зависимости Python
├── data/               # SQLite база данных
│   └── medical_dataset.db
├── audio/              # Сгенерированные аудиофайлы
└── models/             # Скачанные модели Vosk
```

## Быстрый старт

### 1. Инициализация

```bash
# Инициализация базы данных с примерами
python main.py init

# Просмотр статистики
python main.py stats
```

### 2. Генерация аудио

```bash
# Edge TTS (рекомендуется) - требуется интернет
python main.py generate --engine edge

# gTTS (Google) - требуется интернет
python main.py generate --engine gtts

# pyttsx3 (офлайн) - использует системный голос
python main.py generate --engine pyttsx3
```

### 3. Скачивание модели Vosk

```bash
# Компактная русская модель (45 MB) - рекомендуется для начала
python main.py download vosk-model-small-ru-0.22

# Полная русская модель (1.8 GB)
python main.py download vosk-model-ru-0.22
```

### 4. Тестирование модели

```bash
# Тестирование на всех фразах
python main.py test vosk-model-small-ru-0.22

# Тестирование на ограниченном количестве
python main.py test vosk-model-small-ru-0.22 --limit 10
```

### 5. Сравнение моделей

```bash
# Протестируйте несколько моделей
python main.py test vosk-model-small-ru-0.22
python main.py test vosk-model-ru-0.22

# Сравнение
python main.py compare
```

## Структура базы данных

### Таблица `categories`
| Поле | Описание |
|------|---------|
| id | Уникальный ID |
| name | Название категории |
| description | Описание |

### Таблица `phrases`
| Поле | Описание |
|------|---------|
| id | Уникальный ID |
| category_id | ID категории |
| text | Текст фразы |
| speaker | doctor / patient / both |
| complexity | simple / medium / complex |

### Таблица `audio_files`
| Поле | Описание |
|------|---------|
| id | Уникальный ID |
| phrase_id | ID фразы |
| file_path | Путь к файлу |
| tts_engine | Движок синтеза |
| duration_seconds | Длительность |
| sample_rate | Частота дискретизации |

### Таблица `vosk_models`
| Поле | Описание |
|------|---------|
| id | Уникальный ID |
| name | Название модели |
| model_path | Путь к модели |
| is_downloaded | Скачана ли |

### Таблица `recognition_results`
| Поле | Описание |
|------|---------|
| id | Уникальный ID |
| audio_id | ID аудио |
| model_id | ID модели |
| transcription | Распознанный текст |
| confidence | Уверенность (0-1) |
| processing_time_ms | Время обработки |

## Добавление своих фраз

```bash
# Базовая команда
python main.py add "Ваш текст фразы"

# С категорией и типом говорящего
python main.py add "Как давно вы болеете?" --category symptoms --speaker doctor

# Со сложностью
python main.py add "У меня аллергия на пенициллин и амоксициллин" --complexity complex
```

Или через Python API:

```python
from database import MedicalDatasetDB

db = MedicalDatasetDB("data/medical_dataset.db")

# Добавить категорию
cat_id = db.add_category("cardiology", "Кардиология")

# Добавить фразу
phrase_id = db.add_phrase(
    text="У меня повышенное давление",
    category_id=cat_id,
    speaker="patient",
    complexity="medium"
)

# Добавить несколько фраз
db.add_phrases_batch([
    {"text": "Принимайте по одной таблетке утром", "category_id": cat_id},
    {"text": "Измеряйте давление два раза в день", "category_id": cat_id},
])
```

## Программный интерфейс

### Распознавание речи

```python
from vosk_recognizer import VoskRecognizer, VoskModelManager

# Загрузка модели
manager = VoskModelManager("models")
manager.download_model("vosk-model-small-ru-0.22")

# Распознавание
recognizer = VoskRecognizer(
    model_path="models/vosk-model-small-ru-0.22",
    model_name="Small Russian"
)

result = recognizer.recognize_file("audio/phrase_00001.wav")
print(f"Текст: {result.text}")
print(f"Уверенность: {result.confidence}")
print(f"Время: {result.processing_time_ms}ms")
```

### Работа с базой данных

```python
from database import MedicalDatasetDB

db = MedicalDatasetDB("data/medical_dataset.db")

# Получить фразы
phrases = db.get_phrases(category_id=1)

# Получить статистику
stats = db.get_dataset_stats()

# Рассчитать метрики модели
metrics = db.calculate_model_metrics(model_id=1)
print(f"WER: {metrics['avg_wer']}")
print(f"Accuracy: {metrics['accuracy']}")

# Экспорт в CSV
db.export_results_csv(model_id=1, output_path="results.csv")
```

## Метрики качества

- **Accuracy** — процент точных совпадений
- **WER (Word Error Rate)** — отношение ошибок к количеству слов
- **Confidence** — средняя уверенность модели
- **Processing Time** — время распознавания

## Расширение системы

### Добавление нового TTS движка

```python
from tts_generator import TTSGenerator

class MyTTSGenerator(TTSGenerator):
    def generate(self, text: str, filename: str, **kwargs):
        # Ваша реализация
        return filepath, duration
```

### Добавление новой модели Vosk

```python
from vosk_recognizer import VoskModelManager

manager = VoskModelManager()
manager.VOSK_MODELS["my-model"] = {
    "name": "My Custom Model",
    "url": "https://example.com/model.zip",
    "size_mb": 100,
    "description": "Custom model"
}
```

## Требования

- Python 3.8+
- SQLite3 (встроен в Python)
- ffmpeg (для конвертации аудио)

### Установка зависимостей

```bash
pip install -r requirements.txt

# ffmpeg (Ubuntu/Debian)
sudo apt install ffmpeg

# ffmpeg (macOS)
brew install ffmpeg

# ffmpeg (Windows)
# Скачайте с https://ffmpeg.org/download.html
```

## Примеры использования

### Полный цикл тестирования

```bash
# 1. Инициализация
python main.py init

# 2. Генерация аудио (5 фраз для примера)
python main.py generate --limit 5

# 3. Скачивание модели
python main.py download vosk-model-small-ru-0.22

# 4. Тестирование
python main.py test vosk-model-small-ru-0.22

# 5. Просмотр результатов
python main.py stats

# 6. Экспорт
python main.py export 1 --output results.csv
```

## Развитие проекта

Идеи для расширения:

1. **Автоматическая генерация фраз** — использовать LLM для генерации вариаций
2. **Добавление шума** — симуляция разных условий записи
3. **Аугментация аудио** — изменение скорости, тона
4. **Web-интерфейс** — визуализация результатов
5. **API сервер** — REST API для интеграции
6. **Тестирование на реальных записях** — сравнение с синтезированными
