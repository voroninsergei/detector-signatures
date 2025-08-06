"""Модуль для анализа рукописей, имитирующих печатный шрифт.

Печатный (блоковый) почерк отличается отсутствием соединений между
буквами, упрощённой формой элементов и, как правило, большим
соотношением ширины к высоте букв. Этот модуль измеряет коэффициент
связности и соотношения сторон, чтобы определить, принадлежит ли
почерк к печатному типу.
"""

from typing import List
import cv2
import numpy as np
from . import preprocessing, general_features


def analyze_print_like(image_path: str) -> str:
    """Анализирует почерк, выполненный печатными буквами.

    :param image_path: путь к изображению
    :return: текстовый отчёт о признаках печатного письма
    """
    try:
        image = preprocessing.load_image(image_path)
    except FileNotFoundError as exc:
        return str(exc)
    proc = preprocessing.preprocess_image(image)
    connectivity = general_features.compute_connectivity(proc)
    # Оцениваем соотношения сторон символов
    contours, _ = cv2.findContours(proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    aspect_ratios: List[float] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue
        aspect_ratios.append(w / h)
    avg_aspect = float(np.mean(aspect_ratios)) if aspect_ratios else 0.0
    # Эвристика: печатное письмо — низкая связность (<0.1) и среднее
    # соотношение ширины к высоте в пределах 0.5–1.5 (буквы квадратные/высокие)
    if connectivity < 0.1:
        verdict = 'признаки печатного письма (буквы не соединены)'
    else:
        verdict = 'письмо не похоже на печатный шрифт'
    report = (
        f"Анализ печатного письма ({image_path}):\n"
        f"Коэффициент связности: {connectivity:.2f}\n"
        f"Среднее соотношение ширины к высоте букв: {avg_aspect:.2f}\n"
        f"Вывод: {verdict}."
    )
    return report