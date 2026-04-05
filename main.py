#!/usr/bin/env python3
"""
Главный скрипт для тестирования Vosk на медицинских фразах.

Usage:
    python main.py init                    # Инициализация базы данных и примеров
    python main.py generate                # Генерация аудио из фраз
    python main.py download MODEL_NAME      # Скачать модель Vosk
    python main.py test MODEL_NAME         # Тестирование модели
    python main.py compare                 # Сравнение всех моделей
    python main.py stats                   # Статистика по датасету
    python main.py add "phrase text"       # Добавить фразу
    python main.py export MODEL_ID          # Экспорт результатов в CSV
"""

import argparse
import sys
from pathlib import Path

# Добавляем текущую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

from database import MedicalDatasetDB, init_sample_data
from tts_generator import create_tts_generator, generate_dataset_audio
from vosk_recognizer import VoskModelManager, run_batch_recognition, compare_models


def cmd_init(db_path: str = "data/medical_dataset.db"):
    """Инициализация базы данных с примерами."""
    print("Инициализация базы данных...")
    db = MedicalDatasetDB(db_path)
    init_sample_data(db)

    print("\nБаза данных инициализирована!")
    stats = db.get_dataset_stats()
    print(f"\nСтатистика:")
    print(f"  Фраз: {stats['total_phrases']}")
    print(f"  Категорий: {len(stats['by_category'])}")


def cmd_generate(tts_engine: str = "edge", limit: int = None,
                 db_path: str = "data/medical_dataset.db"):
    """Генерация аудиофайлов."""
    print(f"Генерация аудио с движком: {tts_engine}")

    db = MedicalDatasetDB(db_path)
    count = generate_dataset_audio(
        db,
        tts_engine=tts_engine,
        output_dir="audio",
        limit=limit
    )

    print(f"\nСгенерировано {count} аудиофайлов")


def cmd_download(model_name: str = None, models_dir: str = "models"):
    """Скачивание модели Vosk."""
    manager = VoskModelManager(models_dir)

    if model_name is None:
        print("Доступные модели:")
        for key, info in manager.VOSK_MODELS.items():
            status = "✓" if manager.is_model_downloaded(key) else "○"
            print(f"  [{status}] {key}: {info['name']}")
        print("\nИспользование: python main.py download vosk-model-small-ru-0.22")
        return

    path = manager.download_model(model_name)
    print(f"Модель доступна по пути: {path}")


def cmd_test(model_name: str, limit: int = None,
             db_path: str = "data/medical_dataset.db",
             models_dir: str = "models"):
    """Тестирование модели на датасете."""
    manager = VoskModelManager(models_dir)
    model_path = manager.get_model_path(model_name)

    if model_path is None:
        print(f"Модель '{model_name}' не найдена.")
        print("Сначала скачайте её: python main.py download", model_name)
        return

    db = MedicalDatasetDB(db_path)

    # Получаем или создаём запись о модели
    model = db.get_vosk_model_by_name(model_name)
    if model is None:
        model_id = db.add_vosk_model(
            name=model_name,
            model_path=model_path,
            language="ru"
        )
        print(f"Создана запись о модели с ID: {model_id}")
    else:
        model_id = model['id']
        print(f"Используется модель: {model_name} (ID: {model_id})")

    # Запускаем тестирование
    print(f"\nЗапуск распознавания...")
    count = run_batch_recognition(
        db, model_id, model_path, model_name, limit=limit
    )

    # Выводим метрики
    print(f"\n{'='*50}")
    print(f"Результаты для {model_name}:")
    metrics = db.calculate_model_metrics(model_id)
    print(f"  Обработано: {metrics['total_samples']}")
    print(f"  Точность: {metrics['accuracy']*100:.1f}%")
    print(f"  Средняя уверенность: {metrics['avg_confidence']:.2f}")
    print(f"  Среднее время: {metrics['avg_processing_time_ms']:.0f}ms")
    print(f"  WER: {metrics['avg_wer']:.4f}")


def cmd_compare(db_path: str = "data/medical_dataset.db",
                models_dir: str = "models"):
    """Сравнение всех моделей."""
    manager = VoskModelManager(models_dir)
    db = MedicalDatasetDB(db_path)

    models = db.get_vosk_models()
    if not models:
        print("Нет моделей для сравнения.")
        print("Сначала протестируйте модели: python main.py test MODEL_NAME")
        return

    print(f"Сравнение {len(models)} моделей...\n")

    results = []
    for model in models:
        metrics = db.calculate_model_metrics(model['id'])
        if 'error' not in metrics:
            results.append({
                'name': model['name'],
                **metrics
            })

    if not results:
        print("Нет результатов для сравнения.")
        return

    # Сортируем по точности
    results.sort(key=lambda x: x['accuracy'], reverse=True)

    print(f"{'Модель':<30} {'Точность':<12} {'WER':<10} {'Время (ms)':<12} {'Уверенность':<12}")
    print("-" * 80)

    for r in results:
        print(f"{r['name']:<30} "
              f"{r['accuracy']*100:>6.1f}%     "
              f"{r['avg_wer']:.4f}    "
              f"{r['avg_processing_time_ms']:>8.0f}        "
              f"{r['avg_confidence']:.2f}")

    print("\n✓ Лучшая модель:", results[0]['name'] if results else "нет данных")


def cmd_stats(db_path: str = "data/medical_dataset.db"):
    """Показать статистику по датасету."""
    db = MedicalDatasetDB(db_path)
    stats = db.get_dataset_stats()

    print("=" * 50)
    print("СТАТИСТИКА ДАТАСЕТА")
    print("=" * 50)

    print(f"\n📊 Общая информация:")
    print(f"  Фраз в базе: {stats['total_phrases']}")
    print(f"  Аудиофайлов: {stats['total_audio_files']}")
    print(f"  Моделей: {stats['total_models']}")
    print(f"  Результатов: {stats['total_results']}")

    print(f"\n📁 По категориям:")
    for cat in stats['by_category']:
        print(f"  {cat['name']:<20}: {cat['count']} фраз")

    # Статистика по моделям
    models = db.get_vosk_models()
    if models:
        print(f"\n🤖 Установленные модели:")
        for m in models:
            results_count = len(db.get_results_by_model(m['id']))
            print(f"  {m['name']:<30}: {results_count} результатов")
    else:
        print(f"\n🤖 Модели не установлены")


def cmd_add(phrase: str, category: str = None, speaker: str = "both",
            complexity: str = "medium", db_path: str = "data/medical_dataset.db"):
    """Добавить фразу в базу."""
    db = MedicalDatasetDB(db_path)

    category_id = None
    if category:
        category_id = db.add_category(category)

    phrase_id = db.add_phrase(
        text=phrase,
        category_id=category_id,
        speaker=speaker,
        complexity=complexity
    )

    print(f"✓ Фраза добавлена с ID: {phrase_id}")
    print(f"  Текст: {phrase}")
    print(f"  Категория: {category or 'без категории'}")
    print(f"  Говорит: {speaker}")


def cmd_export(model_id: int, output: str = None,
               db_path: str = "data/medical_dataset.db"):
    """Экспорт результатов в CSV."""
    db = MedicalDatasetDB(db_path)

    if output is None:
        output = f"results_model_{model_id}.csv"

    db.export_results_csv(model_id, output)
    print(f"✓ Результаты экспортированы: {output}")


def cmd_list_models(models_dir: str = "models"):
    """Список доступных и установленных моделей."""
    manager = VoskModelManager(models_dir)

    print("Доступные модели Vosk:")
    print("-" * 60)

    for key, info in manager.VOSK_MODELS.items():
        status = "✓ УСТАНОВЛЕНА" if manager.is_model_downloaded(key) else "○ Требуется скачать"
        size = f"{info['size_mb']} MB"

        print(f"\n{key}")
        print(f"  Название: {info['name']}")
        print(f"  Размер: {size}")
        print(f"  Описание: {info['description']}")
        print(f"  Статус: {status}")


def main():
    parser = argparse.ArgumentParser(
        description="Тестирование Vosk на медицинских фразах",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main.py init                      # Инициализация
  python main.py generate                  # Генерация аудио
  python main.py download vosk-model-small-ru-0.22  # Скачать модель
  python main.py test vosk-model-small-ru-0.22      # Тестирование
  python main.py compare                   # Сравнение моделей
  python main.py stats                     # Статистика
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Команды')

    # init
    subparsers.add_parser('init', help='Инициализация базы данных')

    # generate
    gen_parser = subparsers.add_parser('generate', help='Генерация аудио')
    gen_parser.add_argument('--engine', '-e', default='edge',
                            choices=['gtts', 'edge', 'pyttsx3'],
                            help='TTS движок')
    gen_parser.add_argument('--limit', '-l', type=int, default=None,
                            help='Лимит фраз')

    # download
    dl_parser = subparsers.add_parser('download', help='Скачать модель')
    dl_parser.add_argument('model', nargs='?', default=None,
                           help='Название модели')

    # test
    test_parser = subparsers.add_parser('test', help='Тестирование модели')
    test_parser.add_argument('model', help='Название модели')
    test_parser.add_argument('--limit', '-l', type=int, default=None,
                              help='Лимит фраз')

    # compare
    subparsers.add_parser('compare', help='Сравнение моделей')

    # stats
    subparsers.add_parser('stats', help='Статистика датасета')

    # add
    add_parser = subparsers.add_parser('add', help='Добавить фразу')
    add_parser.add_argument('phrase', help='Текст фразы')
    add_parser.add_argument('--category', '-c', default=None,
                            help='Категория')
    add_parser.add_argument('--speaker', '-s', default='both',
                            choices=['doctor', 'patient', 'both'],
                            help='Кто говорит')
    add_parser.add_argument('--complexity', default='medium',
                            choices=['simple', 'medium', 'complex'],
                            help='Сложность')

    # export
    exp_parser = subparsers.add_parser('export', help='Экспорт результатов')
    exp_parser.add_argument('model_id', type=int, help='ID модели')
    exp_parser.add_argument('--output', '-o', default=None,
                            help='Путь к CSV файлу')

    # list-models
    subparsers.add_parser('list-models', help='Список моделей')

    args = parser.parse_args()

    # Выполняем команду
    if args.command == 'init':
        cmd_init()
    elif args.command == 'generate':
        cmd_generate(args.engine, args.limit)
    elif args.command == 'download':
        cmd_download(args.model)
    elif args.command == 'test':
        cmd_test(args.model, args.limit)
    elif args.command == 'compare':
        cmd_compare()
    elif args.command == 'stats':
        cmd_stats()
    elif args.command == 'add':
        cmd_add(args.phrase, args.category, args.speaker, args.complexity)
    elif args.command == 'export':
        cmd_export(args.model_id, args.output)
    elif args.command == 'list-models':
        cmd_list_models()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
