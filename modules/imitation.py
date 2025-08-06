"""Модуль для выявления подражания (имитации) чужому почерку.

Подражание чужому письму часто сопровождается замедлением скорости
письма и большей дрожанием линий. Простая метрика — компактность
контуров (отношение периметра к площади): у «дрожащих» штрихов это
отношение выше. Здесь мы вычисляем среднюю компактность и делаем
предположение о возможной имитации.
"""

from typing import List
import cv2
import numpy as np
from . import preprocessing


def analyze_imitation(image_path: str) -> str:
    """Оценивает признаки подражания чужому почерку.

    :param image_path: путь к изображению
    :return: текстовый отчёт о вероятности подражания
    """
    try:
        image = preprocessing.load_image(image_path)
    except FileNotFoundError as exc:
        return str(exc)
    proc = preprocessing.preprocess_image(image)
    contours, _ = cv2.findContours(proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    compactness_values: List[float] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if area == 0:
            continue
        compactness = (perimeter ** 2) / (4 * np.pi * area)
        compactness_values.append(compactness)
    if not compactness_values:
        return f"На изображении {image_path} не найдено элементов для анализа имитации."
    mean_compactness = float(np.mean(compactness_values))
    # Оценка: высокие значения компактности (>10) могут указывать на дрожание
    if mean_compactness > 10:
        verdict = 'признаки подражания (возможно, письмо обведено/сильно замедлено)'
    elif mean_compactness > 6:
        verdict = 'возможны признаки подражания'
    else:
        verdict = 'признаков подражания не обнаружено'
    report = (
        f"Анализ подражания ({image_path}):\n"
        f"Средняя компактность штрихов: {mean_compactness:.1f}\n"
        f"Вывод: {verdict}."
    )
    return report