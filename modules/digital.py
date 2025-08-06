"""Модуль анализа цифрового почерка.

Этот модуль предназначен для обработки цифровых записей (цифр, числовых
таблиц, дат). Используются детерминистские методы выделения цифр как
связных компонентов и оценка их характеристик: размера, пропорций,
количества и интервалов между ними.
"""

from typing import List
import cv2
import numpy as np
from . import preprocessing, general_features


def analyze_digits(image_path: str) -> str:
    """Проводит анализ цифрового почерка.

    Изображение бинаризуется, выделяются контуры. Для каждой
    предполагаемой цифры оценивается высота, ширина и пропорции. По
    совокупности вычисляется средний размер, разброс, соотношение
    сторон и количество цифр. Используются функции общего модуля для
    вычисления интервалов и наклона.

    :param image_path: путь к изображению
    :return: текстовый отчёт о цифровом почерке
    """
    try:
        image = preprocessing.load_image(image_path)
    except FileNotFoundError as exc:
        return str(exc)
    proc = preprocessing.preprocess_image(image)
    # Находим контуры
    contours, _ = cv2.findContours(proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_heights: List[int] = []
    aspect_ratios: List[float] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20:
            continue  # пропускаем шум
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue
        aspect = w / h
        # Ограничиваем допустимые пропорции цифр
        if 0.2 < aspect < 1.5:
            digit_heights.append(h)
            aspect_ratios.append(aspect)
    if not digit_heights:
        return (
            f"Цифровых компонентов не обнаружено на изображении: {image_path}. "
            "Убедитесь, что файл содержит цифры."
        )
    avg_height = float(np.mean(digit_heights))
    std_height = float(np.std(digit_heights))
    avg_aspect = float(np.mean(aspect_ratios))
    count_digits = len(digit_heights)
    spacing_mean, spacing_std = general_features.compute_spacing(proc)
    slant = general_features.compute_slant(proc)
    report_lines = [
        f"Анализ цифрового почерка ({image_path})",
        f"Количество цифр: {count_digits}",
        f"Средняя высота цифры: {avg_height:.1f} пикселей",
        f"Разброс высоты: {std_height:.1f} пикселей",
        f"Среднее соотношение ширина/высота: {avg_aspect:.2f}",
        f"Средний интервал между цифрами: {spacing_mean:.1f} пикселей",
        f"Наклон цифр: {slant:.1f}°"
    ]
    return '\n'.join(report_lines)