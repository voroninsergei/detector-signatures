"""Модуль для выявления умышленно изменённого (маскированного) почерка.

В умышленно изменённом письме признаки сильно варьируются: автор
старается изменить размер и наклон букв, добавляет нехарактерные
элементы, замедляет письмо. Мы используем простые статистические
показатели — разброс размеров, интервалов и наклона — чтобы оценить
вероятность маскировки.
"""

from typing import List
import numpy as np
from . import preprocessing, general_features


def analyze_intentional_change(image_path: str) -> str:
    """Анализирует вероятность умышленно изменённого почерка.

    :param image_path: путь к файлу изображения
    :return: отчёт со статистикой и выводом о маскировке
    """
    try:
        image = preprocessing.load_image(image_path)
    except FileNotFoundError as exc:
        return str(exc)
    proc = preprocessing.preprocess_image(image)
    lines = preprocessing.segment_text(proc)
    if not lines:
        return (
            f"Не удалось обнаружить текст на изображении {image_path} для анализа маскировки."
        )
    sizes: List[float] = []
    spacings: List[float] = []
    slants: List[float] = []
    for line_img in lines:
        size, _ = general_features.compute_letter_sizes(line_img)
        spacing, _ = general_features.compute_spacing(line_img)
        slant = general_features.compute_slant(line_img)
        sizes.append(size)
        spacings.append(spacing)
        slants.append(slant)
    # Вычисляем разбросы
    size_std = float(np.std(np.array(sizes)))
    spacing_std = float(np.std(np.array(spacings)))
    slant_std = float(np.std(np.array(slants)))
    variation = size_std + spacing_std + slant_std
    # Пороговые значения эмпирические; при тестировании могут быть уточнены
    if variation > 20:
        verdict = 'выраженные признаки маскировки почерка'
    elif variation > 10:
        verdict = 'возможны признаки маскировки'
    else:
        verdict = 'признаков маскировки не обнаружено'
    report = (
        f"Анализ маскированного почерка ({image_path}):\n"
        f"Стандартное отклонение размера строк: {size_std:.1f}\n"
        f"Стандартное отклонение интервалов: {spacing_std:.1f}\n"
        f"Стандартное отклонение наклона: {slant_std:.1f}\n"
        f"Суммарная вариативность признаков: {variation:.1f}\n"
        f"Вывод: {verdict}."
    )
    return report