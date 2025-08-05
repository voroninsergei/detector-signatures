"""Модуль для извлечения общих признаков почерка (этап 3).

Здесь реализуются функции для количественного определения основных
характеристик почерка без привлечения методов машинного обучения.
"""

from typing import Dict, List, Tuple
import cv2
import numpy as np


def _find_components(binary_image: np.ndarray, min_area: int = 10) -> List[np.ndarray]:
    """Выделяет контуры (связные компоненты) на бинарном изображении.

    Функция использует `cv2.findContours` и фильтрует очень маленькие
    компоненты по площади (скорее всего это шум). Возвращает список
    контуров.

    :param binary_image: бинарное изображение (буквы — белые пиксели)
    :param min_area: минимальная площадь компонента для включения
    :return: список контуров
    """
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    return filtered


def compute_letter_sizes(binary_image: np.ndarray) -> Tuple[float, float]:
    """Вычисляет среднюю и стандартную высоту буквенных элементов.

    Для каждой связной компоненты рассчитывается высота ограничивающего
    прямоугольника. Рассчитываем среднее и стандартное отклонение по всем
    элементам. Это даёт представление о размере письма и вариативности.

    :param binary_image: бинарное изображение (буквы — белые пиксели)
    :return: (средняя высота, стандартное отклонение)
    """
    contours = _find_components(binary_image)
    heights = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        heights.append(h)
    if not heights:
        return 0.0, 0.0
    heights_arr = np.array(heights)
    return float(np.mean(heights_arr)), float(np.std(heights_arr))


def compute_spacing(binary_image: np.ndarray) -> Tuple[float, float]:
    """Оценивает средние расстояния между компонентами в строке.

    Рассматривает каждый ряд пикселей и находит интервалы между
    компонентами в этой строке. Возвращает среднее расстояние и
    стандартное отклонение.

    :param binary_image: бинарное изображение (буквы — белые пиксели)
    :return: (среднее расстояние, стандартное отклонение)
    """
    # Находим проекции на горизонтальную ось
    rows_sum = np.sum(binary_image // 255, axis=1)
    # Определяем строки с текстом
    threshold = 0.1 * rows_sum.max() if rows_sum.max() > 0 else 0
    text_rows = np.where(rows_sum > threshold)[0]
    if text_rows.size == 0:
        return 0.0, 0.0
    spacings: List[int] = []
    # Для каждой строки текста вычисляем интервалы между белыми блоками
    for row in text_rows:
        line = binary_image[row]
        positions = np.where(line > 0)[0]
        if positions.size == 0:
            continue
        # Разрывы между соседними белыми пикселями больше 1 означают интервал
        gaps = np.diff(positions)
        # Считаем только большие промежутки (более 1 пикселя)
        large_gaps = gaps[gaps > 1] - 1
        spacings.extend(large_gaps.tolist())
    if not spacings:
        return 0.0, 0.0
    arr = np.array(spacings)
    return float(np.mean(arr)), float(np.std(arr))


def compute_slant(binary_image: np.ndarray) -> float:
    """Вычисляет средний угол наклона компонентов.

    Для каждой компоненты строится эллипс (`cv2.fitEllipse`), если она
    достаточного размера. Возвращается средний угол наклона эллипсов
    относительно вертикали (в градусах). Правый наклон имеет положительный
    угол, левый — отрицательный.

    :param binary_image: бинарное изображение (буквы — белые пиксели)
    :return: средний угол наклона (градусы)
    """
    contours = _find_components(binary_image)
    angles = []
    for cnt in contours:
        if len(cnt) < 5:
            continue  # fitEllipse требует минимум 5 точек
        ellipse = cv2.fitEllipse(cnt)
        angle = ellipse[2]  # угол относительно горизонтали (0-180)
        # Преобразуем в диапазон [-90, 90]
        if angle > 90:
            angle -= 180
        angles.append(angle)
    if not angles:
        return 0.0
    return float(np.mean(angles))


def compute_connectivity(binary_image: np.ndarray) -> float:
    """Вычисляет коэффициент связности письма.

    Сегментируем по строкам, затем анализируем горизонтальные интервалы
    между компонентами. Если промежуток меньше среднего размера компоненты,
    считаем, что буквенная пара связана (письмо без разрыва). Количество
    таких соединений делим на общее количество возможных соединений.

    :param binary_image: бинарное изображение (буквы — белые пиксели)
    :return: коэффициент связности (0.0–1.0)
    """
    # Проекция строк
    rows_sum = np.sum(binary_image // 255, axis=1)
    threshold = 0.1 * rows_sum.max() if rows_sum.max() > 0 else 0
    text_rows = np.where(rows_sum > threshold)[0]
    if text_rows.size == 0:
        return 0.0
    # средняя высота компоненты
    _, h_std = compute_letter_sizes(binary_image)
    # Если нет компонентов, вернем 0
    if h_std == 0:
        return 0.0
    connections = 0
    possible = 0
    for row in text_rows:
        line = binary_image[row]
        positions = np.where(line > 0)[0]
        if positions.size == 0:
            continue
        gaps = np.diff(positions)
        # интервал между буквами – у величина > 1
        for gap in gaps:
            possible += 1
            if gap <= 1:
                connections += 1
    if possible == 0:
        return 0.0
    return connections / possible


def assess_skill_level(size_std: float, spacing_std: float, slant_values: List[float]) -> str:
    """Оценивает степень выработанности (навыка) почерка.

    Использует вариативность размеров, интервалов и наклона. Если
    отклонения малы, считается, что почерк выработан (высокий навык).
    Если разброс средний – навык средний. Большой разброс указывает на
    низкий уровень навыка.

    :param size_std: стандартное отклонение высоты букв
    :param spacing_std: стандартное отклонение интервалов
    :param slant_values: список углов наклона для компонентов
    :return: строковое описание: 'высокая', 'средняя', 'низкая'
    """
    # Нормируем величины и суммируем разбросы
    # Устанавливаем эмпирические пороги
    variation = 0.0
    variation += size_std
    variation += spacing_std
    if slant_values:
        slant_std = float(np.std(np.array(slant_values)))
        variation += slant_std
    # Эмпирические пороги могут быть настроены после тестирования
    if variation < 5:
        return 'высокая'
    elif variation < 15:
        return 'средняя'
    else:
        return 'низкая'