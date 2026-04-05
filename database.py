"""
Модуль управления базой данных для тестирования Vosk на медицинских фразах.

Структура базы данных:
- phrases: медицинские фразы для тестирования
- audio_files: сгенерированные аудиофайлы
- model_results: результаты распознавания от разных моделей
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional
import json


class MedicalDatasetDB:
    """Класс для управления базой данных медицинского датасета."""

    def __init__(self, db_path: str = "data/medical_dataset.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Получить соединение с базой данных."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self):
        """Инициализация структуры базы данных."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Таблица категорий медицинских фраз
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS categories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Таблица медицинских фраз
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS phrases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category_id INTEGER,
                    text TEXT NOT NULL,
                    speaker TEXT CHECK(speaker IN ('doctor', 'patient', 'both')) DEFAULT 'both',
                    complexity TEXT CHECK(complexity IN ('simple', 'medium', 'complex')) DEFAULT 'medium',
                    language TEXT DEFAULT 'ru',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (category_id) REFERENCES categories(id)
                )
            """)

            # Таблица аудиофайлов
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audio_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    phrase_id INTEGER NOT NULL,
                    file_path TEXT NOT NULL,
                    tts_engine TEXT NOT NULL,
                    speaker_voice TEXT,
                    sample_rate INTEGER DEFAULT 16000,
                    duration_seconds REAL,
                    file_size_bytes INTEGER,
                    checksum TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (phrase_id) REFERENCES phrases(id)
                )
            """)

            # Таблица моделей Vosk
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vosk_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    model_path TEXT,
                    model_url TEXT,
                    language TEXT DEFAULT 'ru',
                    version TEXT,
                    description TEXT,
                    is_downloaded BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Таблица результатов распознавания
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS recognition_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    audio_id INTEGER NOT NULL,
                    model_id INTEGER NOT NULL,
                    transcription TEXT NOT NULL,
                    confidence REAL,
                    processing_time_ms REAL,
                    word_count INTEGER,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (audio_id) REFERENCES audio_files(id),
                    FOREIGN KEY (model_id) REFERENCES vosk_models(id)
                )
            """)

            # Таблица для метрик сравнения моделей
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_comparison (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER NOT NULL,
                    total_samples INTEGER DEFAULT 0,
                    correct_samples INTEGER DEFAULT 0,
                    accuracy REAL,
                    avg_confidence REAL,
                    avg_processing_time_ms REAL,
                    avg_word_error_rate REAL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES vosk_models(id)
                )
            """)

            # Создаём индексы для быстрого поиска
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_phrases_category ON phrases(category_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audio_phrase ON audio_files(phrase_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_audio ON recognition_results(audio_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_model ON recognition_results(model_id)")

            conn.commit()

    # === Работа с категориями ===

    def add_category(self, name: str, description: str = None) -> int:
        """Добавить категорию медицинских фраз."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO categories (name, description) VALUES (?, ?)",
                (name, description)
            )
            conn.commit()
            cursor.execute("SELECT id FROM categories WHERE name = ?", (name,))
            return cursor.fetchone()["id"]

    def get_categories(self) -> list:
        """Получить все категории."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM categories ORDER BY name")
            return [dict(row) for row in cursor.fetchall()]

    # === Работа с фразами ===

    def add_phrase(self, text: str, category_id: int = None,
                   speaker: str = "both", complexity: str = "medium") -> int:
        """Добавить медицинскую фразу."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO phrases (text, category_id, speaker, complexity)
                VALUES (?, ?, ?, ?)
            """, (text, category_id, speaker, complexity))
            conn.commit()
            return cursor.lastrowid

    def add_phrases_batch(self, phrases: list) -> int:
        """
        Добавить несколько фраз за раз.

        Args:
            phrases: Список словарей с ключами: text, category_id, speaker, complexity

        Returns:
            Количество добавленных фраз
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            data = [
                (p.get("text"), p.get("category_id"), p.get("speaker", "both"),
                 p.get("complexity", "medium"))
                for p in phrases
            ]
            cursor.executemany("""
                INSERT INTO phrases (text, category_id, speaker, complexity)
                VALUES (?, ?, ?, ?)
            """, data)
            conn.commit()
            return cursor.rowcount

    def get_phrases(self, category_id: int = None,
                    speaker: str = None,
                    complexity: str = None,
                    limit: int = None) -> list:
        """Получить фразы с фильтрацией."""
        query = "SELECT * FROM phrases WHERE 1=1"
        params = []

        if category_id:
            query += " AND category_id = ?"
            params.append(category_id)
        if speaker:
            query += " AND speaker = ?"
            params.append(speaker)
        if complexity:
            query += " AND complexity = ?"
            params.append(complexity)

        query += " ORDER BY id"
        if limit:
            query += f" LIMIT {limit}"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_phrase_by_id(self, phrase_id: int) -> Optional[dict]:
        """Получить фразу по ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM phrases WHERE id = ?", (phrase_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_phrases_without_audio(self, limit: int = None) -> list:
        """Получить фразы, для которых ещё не сгенерированы аудиофайлы."""
        query = """
            SELECT p.* FROM phrases p
            LEFT JOIN audio_files a ON p.id = a.phrase_id
            WHERE a.id IS NULL
            ORDER BY p.id
        """
        if limit:
            query += f" LIMIT {limit}"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]

    def get_phrases_without_results(self, model_id: int, limit: int = None) -> list:
        """Получить фразы с аудио, для которых нет результатов распознавания."""
        query = """
            SELECT DISTINCT p.* FROM phrases p
            JOIN audio_files a ON p.id = a.phrase_id
            LEFT JOIN recognition_results r ON a.id = r.audio_id AND r.model_id = ?
            WHERE r.id IS NULL
            ORDER BY p.id
        """
        if limit:
            query += f" LIMIT {limit}"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (model_id,))
            return [dict(row) for row in cursor.fetchall()]

    # === Работа с аудиофайлами ===

    def add_audio_file(self, phrase_id: int, file_path: str,
                        tts_engine: str, speaker_voice: str = None,
                        sample_rate: int = 16000,
                        duration: float = None,
                        file_size: int = None) -> int:
        """Добавить запись об аудиофайле."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO audio_files
                (phrase_id, file_path, tts_engine, speaker_voice,
                 sample_rate, duration_seconds, file_size_bytes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (phrase_id, file_path, tts_engine, speaker_voice,
                  sample_rate, duration, file_size))
            conn.commit()
            return cursor.lastrowid

    def get_audio_files(self, phrase_id: int = None) -> list:
        """Получить аудиофайлы."""
        query = "SELECT * FROM audio_files WHERE 1=1"
        params = []

        if phrase_id:
            query += " AND phrase_id = ?"
            params.append(phrase_id)

        query += " ORDER BY created_at DESC"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_audio_by_id(self, audio_id: int) -> Optional[dict]:
        """Получить аудиофайл по ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM audio_files WHERE id = ?", (audio_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # === Работа с моделями Vosk ===

    def add_vosk_model(self, name: str, model_path: str = None,
                        model_url: str = None, language: str = "ru",
                        version: str = None, description: str = None) -> int:
        """Добавить модель Vosk."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO vosk_models
                (name, model_path, model_url, language, version, description)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (name, model_path, model_url, language, version, description))
            conn.commit()
            cursor.execute("SELECT id FROM vosk_models WHERE name = ?", (name,))
            return cursor.fetchone()["id"]

    def get_vosk_models(self) -> list:
        """Получить все модели Vosk."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM vosk_models ORDER BY name")
            return [dict(row) for row in cursor.fetchall()]

    def get_vosk_model_by_name(self, name: str) -> Optional[dict]:
        """Получить модель по имени."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM vosk_models WHERE name = ?", (name,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def mark_model_downloaded(self, model_id: int):
        """Отметить модель как скачанную."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE vosk_models SET is_downloaded = 1 WHERE id = ?",
                (model_id,)
            )
            conn.commit()

    # === Работа с результатами распознавания ===

    def add_recognition_result(self, audio_id: int, model_id: int,
                                transcription: str,
                                confidence: float = None,
                                processing_time_ms: float = None,
                                error: str = None) -> int:
        """Добавить результат распознавания."""
        word_count = len(transcription.split()) if transcription else 0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO recognition_results
                (audio_id, model_id, transcription, confidence,
                 processing_time_ms, word_count, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (audio_id, model_id, transcription, confidence,
                  processing_time_ms, word_count, error))
            conn.commit()
            return cursor.lastrowid

    def get_results_by_model(self, model_id: int) -> list:
        """Получить все результаты для модели."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT r.*, a.file_path, p.text as original_text, p.id as phrase_id
                FROM recognition_results r
                JOIN audio_files a ON r.audio_id = a.id
                JOIN phrases p ON a.phrase_id = p.id
                WHERE r.model_id = ?
                ORDER BY r.created_at DESC
            """, (model_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_results_comparison(self, audio_id: int) -> list:
        """Получить сравнение результатов разных моделей для одного аудио."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT r.*, m.name as model_name
                FROM recognition_results r
                JOIN vosk_models m ON r.model_id = m.id
                WHERE r.audio_id = ?
                ORDER BY m.name
            """, (audio_id,))
            return [dict(row) for row in cursor.fetchall()]

    # === Аналитика и метрики ===

    def calculate_model_metrics(self, model_id: int) -> dict:
        """
        Рассчитать метрики качества модели.

        WER (Word Error Rate) вычисляется как:
        (S + D + I) / N, где
        S - количество замен,
        D - количество удалений,
        I - количество вставок,
        N - количество слов в эталоне
        """
        import Levenshtein

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Получаем все результаты с оригинальными фразами
            cursor.execute("""
                SELECT r.transcription, p.text as original, r.confidence,
                       r.processing_time_ms
                FROM recognition_results r
                JOIN audio_files a ON r.audio_id = a.id
                JOIN phrases p ON a.phrase_id = p.id
                WHERE r.model_id = ? AND r.error_message IS NULL
            """, (model_id,))

            results = cursor.fetchall()

            if not results:
                return {"error": "No results for this model"}

            total = len(results)
            correct = 0
            total_confidence = 0
            total_time = 0
            total_wer = 0

            for row in results:
                original_words = row["original"].lower().split()
                transcribed_words = row["transcription"].lower().split()

                # Упрощённый WER (можно использовать любую библиотеку)
                # Здесь используем расстояние Левенштейна для слов
                try:
                    wer = Levenshtein.distance(
                        ' '.join(original_words),
                        ' '.join(transcribed_words)
                    ) / max(len(original_words), 1)
                except:
                    wer = 1.0

                total_wer += wer
                total_confidence += row["confidence"] or 0
                total_time += row["processing_time_ms"] or 0

                # Точное совпадение (после нормализации)
                if self._normalize_text(row["transcription"]) == \
                   self._normalize_text(row["original"]):
                    correct += 1

            avg_confidence = total_confidence / total if total > 0 else 0
            avg_time = total_time / total if total > 0 else 0
            avg_wer = total_wer / total if total > 0 else 0
            accuracy = correct / total if total > 0 else 0

            return {
                "model_id": model_id,
                "total_samples": total,
                "correct_samples": correct,
                "accuracy": accuracy,
                "avg_confidence": avg_confidence,
                "avg_processing_time_ms": avg_time,
                "avg_wer": avg_wer
            }

    def _normalize_text(self, text: str) -> str:
        """Нормализация текста для сравнения."""
        import re
        # Удаляем пунктуацию и приводим к нижнему регистру
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Удаляем лишние пробелы
        text = ' '.join(text.split())
        return text

    def get_dataset_stats(self) -> dict:
        """Получить статистику по датасету."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            stats = {}

            # Количество фраз
            cursor.execute("SELECT COUNT(*) as count FROM phrases")
            stats["total_phrases"] = cursor.fetchone()["count"]

            # По категориям
            cursor.execute("""
                SELECT c.name, COUNT(p.id) as count
                FROM categories c
                LEFT JOIN phrases p ON c.id = p.category_id
                GROUP BY c.id
            """)
            stats["by_category"] = [dict(row) for row in cursor.fetchall()]

            # Количество аудиофайлов
            cursor.execute("SELECT COUNT(*) as count FROM audio_files")
            stats["total_audio_files"] = cursor.fetchone()["count"]

            # Количество моделей
            cursor.execute("SELECT COUNT(*) as count FROM vosk_models")
            stats["total_models"] = cursor.fetchone()["count"]

            # Количество результатов распознавания
            cursor.execute("SELECT COUNT(*) as count FROM recognition_results")
            stats["total_results"] = cursor.fetchone()["count"]

            return stats

    def export_results_csv(self, model_id: int, output_path: str):
        """Экспорт результатов в CSV."""
        import csv

        results = self.get_results_by_model(model_id)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'phrase_id', 'original_text', 'transcription',
                'confidence', 'processing_time_ms', 'wer'
            ])

            for r in results:
                original = r['original_text']
                transcribed = r['transcription']

                # Простой WER
                try:
                    import Levenshtein
                    wer = Levenshtein.distance(original.lower(), transcribed.lower()) / \
                          max(len(original.split()), 1)
                except:
                    wer = 0

                writer.writerow([
                    r['phrase_id'],
                    original,
                    transcribed,
                    r['confidence'],
                    r['processing_time_ms'],
                    f"{wer:.4f}"
                ])


# === Утилиты для работы с базой ===

def init_sample_data(db: MedicalDatasetDB):
    """Инициализация примера данных для тестирования."""

    # Добавляем категории
    categories = [
        ("anamnesis", "Сбор анамнеза"),
        ("symptoms", "Описание симптомов"),
        ("diagnosis", "Диагностика и диагнозы"),
        ("treatment", "Назначение лечения"),
        ("medications", "Лекарственные препараты"),
        ("procedures", "Медицинские процедуры"),
        ("instructions", "Рекомендации пациенту"),
    ]

    for name, desc in categories:
        db.add_category(name, desc)

    # Примеры фраз для разных категорий
    sample_phrases = [
        # Анамнез
        {"text": "Как давно вы чувствуете себя плохо?", "category": "anamnesis", "speaker": "doctor"},
        {"text": "Болит голова уже третий день", "category": "anamnesis", "speaker": "patient"},
        {"text": "Были ли у вас аллергические реакции на лекарства?", "category": "anamnesis", "speaker": "doctor"},
        {"text": "У меня аллергия на пенициллин", "category": "anamnesis", "speaker": "patient"},
        {"text": "Расскажите подробнее о ваших жалобах", "category": "anamnesis", "speaker": "doctor"},

        # Симптомы
        {"text": "Где именно вы чувствуете боль?", "category": "symptoms", "speaker": "doctor"},
        {"text": "Боль в правом подреберье", "category": "symptoms", "speaker": "patient"},
        {"text": "Как сильно вы оцениваете боль по шкале от одного до десяти?", "category": "symptoms", "speaker": "doctor"},
        {"text": "Примерно на семь баллов", "category": "symptoms", "speaker": "patient"},
        {"text": "Есть ли у вас температура?", "category": "symptoms", "speaker": "doctor"},
        {"text": "Да, сегодня утром было тридцать восемь и два", "category": "symptoms", "speaker": "patient"},

        # Диагностика
        {"text": "Мне нужно вас осмотреть и назначить анализы", "category": "diagnosis", "speaker": "doctor"},
        {"text": "Нужно сдать общий анализ крови и мочи", "category": "diagnosis", "speaker": "doctor"},
        {"text": "Также сделаем ультразвуковое исследование брюшной полости", "category": "diagnosis", "speaker": "doctor"},
        {"text": "По результатам анализов можно будет поставить точный диагноз", "category": "diagnosis", "speaker": "doctor"},

        # Лечение
        {"text": "Я назначу вам курс антибиотиков", "category": "treatment", "speaker": "doctor"},
        {"text": "Принимайте по одной таблетке три раза в день", "category": "treatment", "speaker": "doctor"},
        {"text": "Обязательно пейте много воды", "category": "treatment", "speaker": "doctor"},
        {"text": "Через неделю приходите на повторный приём", "category": "treatment", "speaker": "doctor"},

        # Лекарства
        {"text": "Амоксициллин пятьсот миллиграмм", "category": "medications", "speaker": "doctor"},
        {"text": "Принимайте до еды за тридцать минут", "category": "medications", "speaker": "doctor"},
        {"text": "Если появится тошнота, приём можно перенести на после еды", "category": "medications", "speaker": "doctor"},

        # Процедуры
        {"text": "Сейчас я сделаю вам укол", "category": "procedures", "speaker": "doctor"},
        {"text": "Расслабьте плечо, будет небольно", "category": "procedures", "speaker": "doctor"},
        {"text": "Измерю вам артериальное давление", "category": "procedures", "speaker": "doctor"},
        {"text": "Давление сто двадцать на восемьдесят, это нормально", "category": "procedures", "speaker": "doctor"},

        # Рекомендации
        {"text": "Соблюдайте постельный режим в течение трёх дней", "category": "instructions", "speaker": "doctor"},
        {"text": "Исключите жирную и острую пищу", "category": "instructions", "speaker": "doctor"},
        {"text": "Если состояние ухудшится, вызывайте скорую помощь", "category": "instructions", "speaker": "doctor"},
        {"text": "Выздоравливайте!", "category": "instructions", "speaker": "doctor"},
    ]

    categories_map = {c["name"]: c["id"] for c in db.get_categories()}

    for phrase in sample_phrases:
        db.add_phrase(
            text=phrase["text"],
            category_id=categories_map.get(phrase["category"]),
            speaker=phrase["speaker"],
            complexity="medium"
        )

    print(f"Добавлено {len(sample_phrases)} примеров фраз")


if __name__ == "__main__":
    # Тестирование базы данных
    db = MedicalDatasetDB("data/medical_dataset.db")
    init_sample_data(db)

    print("\nСтатистика датасета:")
    stats = db.get_dataset_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nКатегории:")
    for cat in db.get_categories():
        print(f"  - {cat['name']}: {cat['description']}")
