"""Модуль для анализа письма в необычных условиях.

Необычные условия (неудобная поза, плохое освещение, усталость, болезнь) часто
приводят к нарушению привычного ритма письма. Это может проявляться в
изменении наклона, увеличении разброса размеров и интервалов, а также
повышенной дрожжевитости линий. Простая эвристика: высокий разброс
признаков при отсутствии признаков маскировки может указывать на
необычные условия.
"""

from typing import List
import numpy as np
from . import preprocessing, general_features


def analyze_unusual_conditions(image_path: str) -> str:
    """Оценивает, написан ли текст в необычных условиях.

    :param image_path: путь к изображению
    :return: текстовый отчёт о вероятности необычных условий
    """
    try:
        image = preprocessing.load_image(image_path)
    except FileNotFoundError as exc:
        return str(exc)
    proc = preprocessing.preprocess_image(image)
    lines = preprocessing.segment_text(proc)
    if not lines:
        return (
            f"Не удалось найти текст на изображении {image_path} для анализа условий."
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
    # Расчет разбросов
    size_std = float(np.std(np.array(sizes)))
    spacing_std = float(np.std(np.array(spacings)))
    slant_std = float(np.std(np.array(slants)))
    variability = size_std + spacing_std + slant_std
    # Эвристический порог для необычных условий
    if variability > 15:
        verdict = 'вероятны необычные условия письма (большая вариативность)'
    elif variability > 8:
        verdict = 'возможно влияние условий на письмо'
    else:
        verdict = 'признаки необычных условий не обнаружены'
    report = (
        f"Анализ условий письма ({image_path}):\n"
        f"Стандартное отклонение размеров: {size_std:.1f}\n"
        f"Стандартное отклонение интервалов: {spacing_std:.1f}\n"
        f"Стандартное отклонение наклона: {slant_std:.1f}\n"
        f"Суммарная вариативность: {variability:.1f}\n"
        f"Вывод: {verdict}."
    )
    return report