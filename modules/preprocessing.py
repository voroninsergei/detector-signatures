"""Функции предварительной обработки изображений.

На последующих этапах будет реализована полноценная предобработка: чтение файлов,
приведение к оттенкам серого, бинаризация (например, методом Отсу), фильтрация
шума, коррекция наклона и сегментация текста. Пока здесь располагаются
заглушки, возвращающие неизменённый объект.
"""

from typing import List, Tuple
import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Загружает изображение из файла в формате BGR.

    Если файл не удалось открыть, возбуждает исключение.

    :param path: путь к файлу изображения
    :return: изображение в виде массива NumPy
    """
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {path}")
    return image


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Выполняет базовую предварительную обработку изображения.

    * Перевод в оттенки серого;
    * Размытие для подавления шума;
    * Бинаризация методом Отсу;
    * Инверсия, чтобы текст был белым на чёрном фоне (удобнее для поиска контуров).

    :param image: исходное изображение
    :return: бинаризированное изображение
    """
    # Переводим изображение в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Гауссово размытие для сглаживания
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Бинаризация методом Отсу
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Инвертируем изображение: буквы белые, фон чёрный
    inverted = 255 - binary
    return inverted


def segment_text(image: np.ndarray, *, line_threshold: float = 0.1) -> List[np.ndarray]:
    """Сегментирует бинарное изображение на строки текста.

    Использует горизонтальные проекции: суммирует пиксели по строкам и
    определяет границы строк по пробелам. Это простейший метод, который
    работает для аккуратно отсканированных документов.

    :param image: бинарное изображение (желательно уже инвертированное)
    :param line_threshold: доля от максимальной суммы по строке, ниже которой
        принимается решение об интервале между строками
    :return: список изображений строк
    """
    # Вычисляем горизонтальную проекцию – количество белых пикселей в каждой строке
    projection = np.sum(image // 255, axis=1)
    max_val = projection.max()
    if max_val == 0:
        # Нет текста
        return []

    threshold_value = max_val * line_threshold
    lines: List[np.ndarray] = []
    in_line = False
    start_row = 0
    for i, val in enumerate(projection):
        if val > threshold_value and not in_line:
            # Начинается строка
            in_line = True
            start_row = i
        elif val <= threshold_value and in_line:
            # Заканчивается строка
            end_row = i
            line_img = image[start_row:end_row, :]
            # Отсекаем пустые строки
            if line_img.shape[0] > 2:
                lines.append(line_img)
            in_line = False
    # Обработка последней строки, если файл заканчивается текстом
    if in_line:
        end_row = len(projection)
        line_img = image[start_row:end_row, :]
        if line_img.shape[0] > 2:
            lines.append(line_img)
    return lines