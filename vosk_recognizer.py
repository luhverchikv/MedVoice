"""
Модуль распознавания речи с использованием Vosk.

Vosk - это открытая библиотека для распознавания речи,
поддерживающая оффлайн-режим и русский язык.
"""

import json
import time
import wave
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass
import numpy as np


@dataclass
class RecognitionResult:
    """Результат распознавания."""
    text: str
    confidence: float
    words: List[dict]
    processing_time_ms: float
    model_name: str


class VoskRecognizer:
    """Класс для распознавания речи с помощью Vosk."""

    def __init__(self, model_path: str, model_name: str = "unknown"):
        """
        Инициализация распознавателя.

        Args:
            model_path: Путь к модели Vosk
            model_name: Название модели для отображения
        """
        self.model_path = Path(model_path)
        self.model_name = model_name
        self._model = None
        self._recognizer = None

    def _load_model(self):
        """Загрузка модели Vosk."""
        if self._model is None:
            try:
                from vosk import Model, KaldiRecognizer
                print(f"Загрузка модели Vosk: {self.model_name}")
                self._model = Model(str(self.model_path))
                self._recognizer = KaldiRecognizer(self._model, 16000)
                print(f"Модель загружена успешно")
            except ImportError:
                raise ImportError(
                    "Vosk не установлен. Установите: pip install vosk"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Не удалось загрузить модель: {e}. "
                    f"Проверьте путь: {self.model_path}"
                )

    def recognize_file(self, audio_path: str) -> RecognitionResult:
        """
        Распознаёт речь из аудиофайла.

        Args:
            audio_path: Путь к аудиофайлу (wav, mp3 и т.д.)

        Returns:
            RecognitionResult: Результат распознавания
        """
        self._load_model()

        audio_path = Path(audio_path)
        start_time = time.time()

        try:
            # Vosk работает с WAV файлами, поэтому конвертируем если нужно
            if audio_path.suffix.lower() != '.wav':
                import subprocess
                temp_wav = audio_path.with_suffix('.wav')
                subprocess.run([
                    'ffmpeg', '-i', str(audio_path), '-ar', '16000',
                    '-ac', '1', '-y', str(temp_wav)
                ], capture_output=True, check=True)
                audio_path = temp_wav

            # Читаем WAV файл
            with wave.open(str(audio_path), "rb") as wf:
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()

                # Проверяем формат
                if sample_rate != 16000:
                    raise ValueError(
                        f"Неверная частота дискретизации: {sample_rate}. "
                        f"Требуется 16000 Hz"
                    )

                # Сбрасываем recognizer для нового файла
                self._recognizer.Reset()

                # Читаем и обрабатываем файл частями
                results = []
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break

                    if self._recognizer.AcceptWaveform(data):
                        result = json.loads(self._recognizer.Result())
                        results.append(result)

                # Получаем финальный результат
                final_result = json.loads(self._recognizer.FinalResult())
                results.append(final_result)

            # Обработка результатов
            full_text = ""
            all_words = []
            total_confidence = 0
            count = 0

            for result in results:
                if 'result' in result:
                    all_words.extend(result['result'])
                if 'text' in result and result['text']:
                    full_text += result['text'] + " "

            # Вычисляем среднюю уверенность
            if all_words:
                for word in all_words:
                    total_confidence += word.get('conf', 0)
                    count += 1
                avg_confidence = total_confidence / count if count > 0 else 0
            else:
                # Если нет данных о словах, используем пустую строку
                avg_confidence = 0

            processing_time = (time.time() - start_time) * 1000

            return RecognitionResult(
                text=full_text.strip(),
                confidence=avg_confidence,
                words=all_words,
                processing_time_ms=processing_time,
                model_name=self.model_name
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return RecognitionResult(
                text="",
                confidence=0,
                words=[],
                processing_time_ms=processing_time,
                model_name=self.model_name
            )

    def recognize_bytes(self, audio_data: bytes) -> RecognitionResult:
        """
        Распознаёт речь из байтов аудио.

        Args:
            audio_data: Байты аудио (16-bit PCM, 16kHz, mono)

        Returns:
            RecognitionResult: Результат распознавания
        """
        self._load_model()

        start_time = time.time()

        try:
            self._recognizer.Reset()
            self._recognizer.AcceptWaveform(audio_data)
            result = json.loads(self._recognizer.FinalResult())

            processing_time = (time.time() - start_time) * 1000

            words = result.get('result', [])
            confidence = sum(w.get('conf', 0) for w in words) / len(words) if words else 0

            return RecognitionResult(
                text=result.get('text', ''),
                confidence=confidence,
                words=words,
                processing_time_ms=processing_time,
                model_name=self.model_name
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return RecognitionResult(
                text="",
                confidence=0,
                words=[],
                processing_time_ms=processing_time,
                model_name=self.model_name
            )


class VoskModelManager:
    """Менеджер для управления моделями Vosk."""

    VOSK_MODELS = {
        "vosk-model-small-ru-0.22": {
            "name": "Vosk Small Russian (0.22)",
            "url": "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip",
            "size_mb": 45,
            "description": "Компактная модель для русского языка"
        },
        "vosk-model-ru-0.22": {
            "name": "Vosk Standard Russian (0.22)",
            "url": "https://alphacephei.com/vosk/models/vosk-model-ru-0.22.zip",
            "size_mb": 1800,
            "description": "Стандартная модель для русского языка"
        },
        "vosk-model-small-en-us-0.15": {
            "name": "Vosk Small English (0.15)",
            "url": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
            "size_mb": 40,
            "description": "Компактная модель для английского языка"
        },
        "vosk-model-cn-0.3": {
            "name": "Vosk Chinese (0.3)",
            "url": "https://alphacephei.com/vosk/models/vosk-model-cn-0.3.zip",
            "size_mb": 1500,
            "description": "Модель для китайского языка"
        },
    }

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def list_available_models(self) -> dict:
        """Возвращает список доступных моделей."""
        return self.VOSK_MODELS

    def is_model_downloaded(self, model_name: str) -> bool:
        """Проверяет, скачана ли модель."""
        model_path = self.models_dir / model_name
        return model_path.exists()

    def download_model(self, model_name: str) -> str:
        """
        Скачивает модель Vosk.

        Args:
            model_name: Название модели

        Returns:
            str: Путь к скачанной модели
        """
        if model_name not in self.VOSK_MODELS:
            raise ValueError(
                f"Неизвестная модель: {model_name}. "
                f"Доступные: {list(self.VOSK_MODELS.keys())}"
            )

        model_info = self.VOSK_MODELS[model_name]
        model_path = self.models_dir / model_name

        if model_path.exists():
            print(f"Модель уже существует: {model_path}")
            return str(model_path)

        print(f"Скачивание модели {model_info['name']}...")
        print(f"Размер: ~{model_info['size_mb']} MB")
        print(f"URL: {model_info['url']}")

        import urllib.request
        import zipfile
        import shutil

        zip_path = self.models_dir / f"{model_name}.zip"

        try:
            # Скачиваем
            urllib.request.urlretrieve(model_info['url'], zip_path)

            # Распаковываем
            print("Распаковка...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.models_dir)

            # Удаляем архив
            zip_path.unlink()

            print(f"Модель установлена: {model_path}")
            return str(model_path)

        except Exception as e:
            # Очищаем при ошибке
            if zip_path.exists():
                zip_path.unlink()
            if model_path.exists():
                shutil.rmtree(model_path)
            raise RuntimeError(f"Ошибка при скачивании модели: {e}")

    def get_model_path(self, model_name: str) -> Optional[str]:
        """Получает путь к модели, если она скачана."""
        model_path = self.models_dir / model_name
        if model_path.exists():
            return str(model_path)
        return None


# === Утилиты для тестирования ===

def test_model(model_path: str, audio_path: str, model_name: str = "test") -> dict:
    """
    Тестирует модель на одном аудиофайле.

    Returns:
        dict: Результаты тестирования
    """
    recognizer = VoskRecognizer(model_path, model_name)
    result = recognizer.recognize_file(audio_path)

    return {
        "model": model_name,
        "audio": audio_path,
        "recognized_text": result.text,
        "confidence": result.confidence,
        "processing_time_ms": result.processing_time_ms,
        "word_count": len(result.words)
    }


def run_batch_recognition(db, model_id: int, model_path: str,
                           model_name: str, limit: int = None) -> int:
    """
    Запускает пакетное распознавание для всех аудиофайлов.

    Args:
        db: Экземпляр MedicalDatasetDB
        model_id: ID модели в базе данных
        model_path: Путь к модели Vosk
        model_name: Название модели
        limit: Максимальное количество файлов

    Returns:
        int: Количество обработанных файлов
    """
    recognizer = VoskRecognizer(model_path, model_name)

    phrases = db.get_phrases_without_results(model_id, limit=limit)

    count = 0
    for phrase in phrases:
        # Получаем аудиофайл для фразы
        audio_files = db.get_audio_files(phrase_id=phrase['id'])
        if not audio_files:
            continue

        audio = audio_files[0]  # Берём первый аудиофайл

        try:
            result = recognizer.recognize_file(audio['file_path'])

            # Сохраняем результат в базу
            db.add_recognition_result(
                audio_id=audio['id'],
                model_id=model_id,
                transcription=result.text,
                confidence=result.confidence,
                processing_time_ms=result.processing_time_ms
            )

            count += 1
            print(f"[{count}/{len(phrases)}] {phrase['text'][:50]}...")
            print(f"   → {result.text[:50]}...")
            print(f"   confidence: {result.confidence:.2f}, "
                  f"time: {result.processing_time_ms:.0f}ms")

        except Exception as e:
            print(f"Ошибка при распознавании фразы {phrase['id']}: {e}")

    return count


def compare_models(db, model_ids: list) -> dict:
    """
    Сравнивает результаты нескольких моделей.

    Args:
        db: Экземпляр MedicalDatasetDB
        model_ids: Список ID моделей для сравнения

    Returns:
        dict: Результаты сравнения
    """
    comparison = {}

    for model_id in model_ids:
        metrics = db.calculate_model_metrics(model_id)
        comparison[model_id] = metrics

    return comparison


if __name__ == "__main__":
    print("Тестирование модуля Vosk...")

    # Проверяем установку
    try:
        import vosk
        print("✓ Vosk установлен")
    except ImportError:
        print("✗ Vosk не установлен")
        print("  Установите: pip install vosk")

    # Показываем доступные модели
    manager = VoskModelManager()
    print("\nДоступные модели Vosk:")
    for key, info in manager.VOSK_MODELS.items():
        status = "✓ скачана" if manager.is_model_downloaded(key) else "○ требуется скачать"
        print(f"  {key}: {info['name']} ({info['size_mb']} MB) [{status}]")

    print("\nДля скачивания модели используйте:")
    print("  manager.download_model('vosk-model-small-ru-0.22')")
