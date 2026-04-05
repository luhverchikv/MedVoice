"""
Модуль синтеза речи (Text-to-Speech) для генерации аудио из медицинских фраз.

Поддерживает несколько TTS движков:
- gTTS (Google Translate TTS) - бесплатный, хорошее качество
- pyttsx3 (offline TTS) - работает без интернета
- Edge TTS (Microsoft Edge) - высокое качество, русский язык
"""

import os
import hashlib
import time
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class AudioConfig:
    """Конфигурация для генерации аудио."""
    sample_rate: int = 16000
    channels: int = 1
    bit_depth: int = 16


class TTSGenerator:
    """Базовый класс для генерации речи."""

    def __init__(self, output_dir: str = "audio", config: AudioConfig = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or AudioConfig()

    def generate(self, text: str, filename: str, **kwargs) -> Tuple[str, float]:
        """
        Генерирует аудиофайл из текста.

        Returns:
            Tuple[str, float]: (путь к файлу, длительность в секундах)
        """
        raise NotImplementedError

    def _get_file_checksum(self, filepath: str) -> str:
        """Вычисляет MD5 хеш файла."""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def _get_file_size(self, filepath: str) -> int:
        """Получает размер файла в байтах."""
        return os.path.getsize(filepath)


class gTTSGenerator(TTSGenerator):
    """
    Генератор речи на основе Google Translate TTS.
    Быстрый, хорошее качество, требует интернет.
    """

    def __init__(self, output_dir: str = "audio", lang: str = "ru", **kwargs):
        super().__init__(output_dir, **kwargs)
        self.lang = lang
        self._gtts = None

    def _get_gtts(self):
        if self._gtts is None:
            try:
                from gtts import gTTS
                self._gtts = gTTS
            except ImportError:
                raise ImportError(
                    "gTTS не установлен. Установите: pip install gtts"
                )
        return self._gtts

    def generate(self, text: str, filename: str, **kwargs) -> Tuple[str, float]:
        """Генерирует аудио с помощью gTTS."""
        tts = self._get_gtts()
        lang = kwargs.get('lang', self.lang)

        filepath = self.output_dir / filename

        # Создаём TTS объект и сохраняем
        speech = tts(text=text, lang=lang, slow=False)
        speech.save(str(filepath))

        # gTTS сохраняет в mp3, нужно конвертировать в wav
        if filepath.suffix == '.wav':
            # Конвертируем mp3 в wav
            mp3_path = filepath.with_suffix('.mp3')
            if mp3_path.exists():
                import subprocess
                subprocess.run([
                    'ffmpeg', '-i', str(mp3_path), '-ar',
                    str(self.config.sample_rate), '-ac', str(self.config.channels),
                    str(filepath), '-y'
                ], capture_output=True)
                mp3_path.unlink()  # Удаляем mp3

        # Получаем длительность
        duration = self._get_duration(filepath)

        return str(filepath), duration

    def _get_duration(self, filepath: Path) -> float:
        """Получает длительность аудиофайла."""
        try:
            import soundfile as sf
            data, samplerate = sf.read(str(filepath))
            return len(data) / samplerate
        except:
            # Если не удалось определить, возвращаем примерную
            return len(filepath.read_bytes()) / (self.config.sample_rate * 2)


class EdgeTTSGenerator(TTSGenerator):
    """
    Генератор речи на основе Microsoft Edge TTS.
    Высокое качество, поддержка русского языка, требует интернет.
    """

    VOICE_MAP = {
        "ru": {
            "female": "ru-RU-SvetlanaNeural",
            "male": "ru-RU-DmitryNeural",
        },
        "en": {
            "female": "en-US-JennyNeural",
            "male": "en-US-GuyNeural",
        }
    }

    def __init__(self, output_dir: str = "audio", lang: str = "ru",
                 voice_gender: str = "female", **kwargs):
        super().__init__(output_dir, **kwargs)
        self.lang = lang
        self.voice_gender = voice_gender

    def generate(self, text: str, filename: str, **kwargs) -> Tuple[str, float]:
        """Генерирует аудио с помощью Edge TTS."""
        import edge_tts

        gender = kwargs.get('voice_gender', self.voice_gender)
        lang = kwargs.get('lang', self.lang)

        voice = self.VOICE_MAP.get(lang, self.VOICE_MAP["ru"]).get(
            gender, self.VOICE_MAP["ru"]["female"]
        )

        filepath = self.output_dir / filename

        # Генерируем аудио
        import asyncio

        async def generate_audio():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(filepath))

        asyncio.run(generate_audio())

        # Конвертируем в нужный формат если нужно
        # Edge TTS по умолчанию создаёт mp3, конвертируем в wav
        if filepath.suffix == '.wav':
            mp3_path = filepath.with_suffix('.mp3')
            if not mp3_path.exists():
                mp3_path = filepath
                filepath = filepath.with_suffix('.mp3')
                mp3_path.rename(filepath)

            try:
                import subprocess
                subprocess.run([
                    'ffmpeg', '-i', str(mp3_path), '-ar',
                    str(self.config.sample_rate), '-ac', str(self.config.channels),
                    str(filepath.with_suffix('.wav')), '-y'
                ], capture_output=True, check=True)

                # Удаляем mp3 и возвращаем wav путь
                mp3_path.unlink(missing_ok=True)
                filepath = filepath.with_suffix('.wav')
            except subprocess.CalledProcessError:
                # ffmpeg не найден, возвращаем mp3
                pass

        duration = self._get_duration(filepath)

        return str(filepath), duration

    def _get_duration(self, filepath: Path) -> float:
        """Получает длительность аудиофайла."""
        try:
            import soundfile as sf
            data, samplerate = sf.read(str(filepath))
            return len(data) / samplerate
        except ImportError:
            try:
                import librosa
                duration = librosa.get_duration(filename=str(filepath))
                return duration
            except:
                return 0


class Pyttsx3Generator(TTSGenerator):
    """
    Оффлайн генератор речи с использованием системного TTS.
    Не требует интернета, качество зависит от системы.
    """

    def __init__(self, output_dir: str = "audio", rate: int = 150, **kwargs):
        super().__init__(output_dir, **kwargs)
        self.rate = rate
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            try:
                import pyttsx3
                self._engine = pyttsx3.init()
                self._engine.setProperty('rate', self.rate)
                self._engine.setProperty('volume', 1.0)
            except ImportError:
                raise ImportError(
                    "pyttsx3 не установлен. Установите: pip install pyttsx3"
                )
        return self._engine

    def generate(self, text: str, filename: str, **kwargs) -> Tuple[str, float]:
        """Генерирует аудио с помощью pyttsx3."""
        engine = self._get_engine()
        filepath = self.output_dir / filename

        # Сохраняем в wav
        engine.save_to_file(text, str(filepath))
        engine.runAndWait()

        # Конвертируем в нужную частоту дискретизации
        try:
            import subprocess
            subprocess.run([
                'ffmpeg', '-i', str(filepath), '-ar',
                str(self.config.sample_rate), '-ac', str(self.config.channels),
                str(filepath), '-y'
            ], capture_output=True)
        except FileNotFoundError:
            pass  # ffmpeg не установлен, используем как есть

        duration = self._get_duration(filepath)

        return str(filepath), duration

    def _get_duration(self, filepath: Path) -> float:
        """Получает длительность аудиофайла."""
        try:
            import soundfile as sf
            data, samplerate = sf.read(str(filepath))
            return len(data) / samplerate
        except:
            return 0

    def get_available_voices(self) -> list:
        """Получить список доступных голосов."""
        engine = self._get_engine()
        voices = engine.getProperty('voices')
        return [{"id": v.id, "name": v.name, "languages": v.languages}
                for v in voices]


def create_tts_generator(engine: str = "edge", **kwargs) -> TTSGenerator:
    """
    Фабричная функция для создания TTS генератора.

    Args:
        engine: Тип движка ('gtts', 'edge', 'pyttsx3')
        **kwargs: Дополнительные параметры

    Returns:
        TTSGenerator: Экземпляр генератора
    """
    engines = {
        "gtts": gTTSGenerator,
        "edge": EdgeTTSGenerator,
        "pyttsx3": Pyttsx3Generator,
    }

    if engine not in engines:
        raise ValueError(f"Неизвестный движок: {engine}. "
                         f"Доступные: {list(engines.keys())}")

    return engines[engine](**kwargs)


# === Утилиты для пакетной генерации ===

def generate_dataset_audio(db, tts_engine: str = "edge",
                            output_dir: str = "audio",
                            limit: int = None) -> int:
    """
    Генерирует аудиофайлы для всех фраз в базе данных.

    Args:
        db: Экземпляр MedicalDatasetDB
        tts_engine: Тип TTS движка
        output_dir: Директория для сохранения аудио
        limit: Максимальное количество фраз (None = все)

    Returns:
        int: Количество сгенерированных файлов
    """
    from database import MedicalDatasetDB

    generator = create_tts_generator(tts_engine, output_dir=output_dir)

    phrases = db.get_phrases_without_audio(limit=limit)

    count = 0
    for phrase in phrases:
        try:
            # Формируем имя файла
            filename = f"phrase_{phrase['id']:05d}.wav"

            # Генерируем аудио
            filepath, duration = generator.generate(
                phrase['text'],
                filename,
                lang='ru'
            )

            # Добавляем запись в базу
            file_size = os.path.getsize(filepath)
            db.add_audio_file(
                phrase_id=phrase['id'],
                file_path=filepath,
                tts_engine=tts_engine,
                sample_rate=16000,
                duration=duration,
                file_size=file_size
            )

            count += 1
            print(f"[{count}/{len(phrases)}] Сгенерировано: {filename}")

        except Exception as e:
            print(f"Ошибка при генерации фразы {phrase['id']}: {e}")

    return count


if __name__ == "__main__":
    # Тестирование TTS
    print("Тестирование TTS генераторов...")

    test_text = "Здравствуйте, как вы себя чувствуете?"

    # Тест gTTS
    try:
        print("\n1. Тест gTTS...")
        gen = create_tts_generator("gtts")
        path, duration = gen.generate(test_text, "test_gtts.mp3")
        print(f"   ✓ Сохранено: {path} ({duration:.2f} сек)")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")

    # Тест Edge TTS
    try:
        print("\n2. Тест Edge TTS...")
        gen = create_tts_generator("edge")
        path, duration = gen.generate(test_text, "test_edge.wav")
        print(f"   ✓ Сохранено: {path} ({duration:.2f} сек)")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")

    # Тест pyttsx3
    try:
        print("\n3. Тест pyttsx3...")
        gen = create_tts_generator("pyttsx3")
        path, duration = gen.generate(test_text, "test_pyttsx3.wav")
        print(f"   ✓ Сохранено: {path} ({duration:.2f} сек)")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")

    print("\nГотово!")
