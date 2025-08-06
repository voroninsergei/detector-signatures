"""Модуль сравнительного анализа образцов (этап 4).

Реализует функции для сопоставления двух рукописных изображений по
извлечённым общим признакам. Алгоритм использует детерминистские
метрики: сравниваются размеры букв, интервалы, наклоны и связность.
Индивидуальные (частные) признаки будут учитываться на следующих
этапах.
"""

from typing import Dict, Tuple
from . import preprocessing, general_features
import numpy as np


def _categorize_size(avg_height: float) -> int:
    """Категоризация размера букв: 0 – малый, 1 – средний, 2 – крупный."""
    if avg_height < 15:
        return 0
    elif avg_height < 30:
        return 1
    else:
        return 2


def compare_images(image_path_a: str, image_path_b: str) -> Tuple[float, Dict[str, Tuple[float, float]]]:
    """Сравнивает два рукописных изображения и возвращает оценку сходства.

    Изображения обрабатываются (предобработка, бинаризация), затем
    извлекаются общие признаки. На основе различий вычисляется
    коэффициент сходства (0–1). Также возвращаются подробности о
    различиях по каждому признаку для отчёта.

    :param image_path_a: путь к первому изображению (спорный документ)
    :param image_path_b: путь к второму изображению (образец для сравнения)
    :return: (коэффициент сходства, словарь отличий по признакам)
    """
    # Загружаем и обрабатываем изображения
    img_a = preprocessing.load_image(image_path_a)
    img_b = preprocessing.load_image(image_path_b)
    proc_a = preprocessing.preprocess_image(img_a)
    proc_b = preprocessing.preprocess_image(img_b)
    # Извлекаем общие признаки
    size_a, size_std_a = general_features.compute_letter_sizes(proc_a)
    space_a, space_std_a = general_features.compute_spacing(proc_a)
    slant_a = general_features.compute_slant(proc_a)
    conn_a = general_features.compute_connectivity(proc_a)

    size_b, size_std_b = general_features.compute_letter_sizes(proc_b)
    space_b, space_std_b = general_features.compute_spacing(proc_b)
    slant_b = general_features.compute_slant(proc_b)
    conn_b = general_features.compute_connectivity(proc_b)
    # Сохраняем подробности
    details = {
        'size_avg': (size_a, size_b),
        'spacing_avg': (space_a, space_b),
        'slant': (slant_a, slant_b),
        'connectivity': (conn_a, conn_b)
    }
    # Категоризация размера
    cat_a = _categorize_size(size_a)
    cat_b = _categorize_size(size_b)
    diff_cat = abs(cat_a - cat_b)
    score_size = 1.0 - min(diff_cat / 2.0, 1.0)
    # Разгон (интервал) – нормируем разницу относительно среднего величины
    max_space = max(space_a, space_b, 1e-3)
    diff_space = abs(space_a - space_b) / max_space
    score_spacing = 1.0 - min(diff_space, 1.0)
    # Наклон – нормируем на 90°
    diff_slant = abs(slant_a - slant_b) / 90.0
    score_slant = 1.0 - min(diff_slant, 1.0)
    # Связность – разница абсолютная
    diff_conn = abs(conn_a - conn_b)
    score_conn = 1.0 - min(diff_conn, 1.0)
    # Итоговый коэффициент – среднее
    similarity = (score_size + score_spacing + score_slant + score_conn) / 4.0
    return similarity, details


def interpret_similarity(similarity: float) -> str:
    """Интерпретирует коэффициент сходства в словесную категорию.

    :param similarity: коэффициент (0–1)
    :return: словесный вывод
    """
    if similarity >= 0.75:
        return 'высокая степень сходства (возможно, выполнял один исполнитель)'
    elif similarity >= 0.5:
        return 'умеренная степень сходства (необходимо дополнительное исследование)'
    else:
        return 'низкая степень сходства (вероятно, разные исполнители)'