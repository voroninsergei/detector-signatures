"""Модуль для анализа почерка с разрывом во времени.

Этот модуль предназначен для сравнения почерка одного лица, выполненного
в разные периоды времени (например, с промежутком в несколько лет). 
Анализ помогает понять, какие общие признаки остаются устойчивыми, а какие
изменяются из-за возрастных изменений или условий. Функция может
принимать путь к одному изображению или несколько путей (через запятую)
и анализировать динамику признаков.
"""

from typing import List, Tuple
from . import preprocessing, general_features
import os


def analyze_time_gap(image_paths: str) -> str:
    """Анализ почерка, выполненного с разрывом во времени.

    Если передано несколько путей через запятую, функция посчитает
    показатели для каждого и выведет разницу между первым и последним.

    :param image_paths: путь(и) к изображениям, разделённые запятой
    :return: текстовый отчёт о динамике общих признаков
    """
    # Разбираем список путей
    paths: List[str] = [p.strip() for p in image_paths.split(',') if p.strip()]
    if not paths:
        return "Путь к изображению не указан."
    results: List[Tuple[str, float, float, float, float]] = []  # name, size, spacing, slant, conn
    for path in paths:
        try:
            img = preprocessing.load_image(path)
        except FileNotFoundError:
            return f"Файл {path} не найден."
        proc = preprocessing.preprocess_image(img)
        size, _ = general_features.compute_letter_sizes(proc)
        spacing, _ = general_features.compute_spacing(proc)
        slant = general_features.compute_slant(proc)
        conn = general_features.compute_connectivity(proc)
        results.append((os.path.basename(path), size, spacing, slant, conn))
    lines = ["Анализ почерка с разрывом во времени:"]
    for idx, (name, size, spacing, slant, conn) in enumerate(results):
        lines.append(
            f"Образец {idx+1} ({name}): средний размер {size:.1f} px, "
            f"интервал {spacing:.1f} px, наклон {slant:.1f}°, связность {conn:.2f}"
        )
    if len(results) >= 2:
        first = results[0]
        last = results[-1]
        diff_size = last[1] - first[1]
        diff_spacing = last[2] - first[2]
        diff_slant = last[3] - first[3]
        diff_conn = last[4] - first[4]
        lines.append("\nДинамика изменений (последний - первый образец):")
        lines.append(
            f"Δ размер = {diff_size:.1f} px, Δ интервал = {diff_spacing:.1f} px, "
            f"Δ наклон = {diff_slant:.1f}°, Δ связность = {diff_conn:.2f}"
        )
    return '\n'.join(lines)