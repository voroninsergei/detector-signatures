"""Модуль для анализа почерка, выполненного левой рукой.

Левая рука зачастую приводит к инверсии наклона (влево), меньшей
связности письма и общему замедлению. Простейший анализ будет
определять доминирующий наклон и оценивать связность.
"""

from . import preprocessing, general_features


def analyze_left_hand(image_path: str) -> str:
    """Оценивает, был ли текст написан левой рукой.

    :param image_path: путь к изображению
    :return: отчёт с оценкой вероятности левой руки
    """
    try:
        image = preprocessing.load_image(image_path)
    except FileNotFoundError as exc:
        return str(exc)
    proc = preprocessing.preprocess_image(image)
    slant = general_features.compute_slant(proc)
    connectivity = general_features.compute_connectivity(proc)
    # Простая эвристика: наклон < -10° указывает на левый наклон (вероятно левая рука)
    # При этом низкая связность (<0.3) усиливает подозрение
    if slant < -10 and connectivity < 0.3:
        verdict = 'вероятно, письмо выполнено левой рукой (наклон влево, низкая связность)'
    elif slant < -5:
        verdict = 'возможно письмо левой рукой (наклон влево)'
    else:
        verdict = 'признаков письма левой рукой не обнаружено'
    report = (
        f"Анализ письма левой рукой ({image_path}):\n"
        f"Средний наклон: {slant:.1f}°\n"
        f"Коэффициент связности: {connectivity:.2f}\n"
        f"Вывод: {verdict}."
    )
    return report