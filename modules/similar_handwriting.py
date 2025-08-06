"""Модуль для анализа сходных почерков.

Использует общий модуль сравнительного анализа для оценки степени
сходства между двумя образцами. Служит упрощённой обёрткой над
`comparative.compare_images`.
"""

from . import comparative


def analyze_similarity(image_path_a: str, image_path_b: str) -> str:
    """Сравнивает два изображения и возвращает текстовый отчёт.

    :param image_path_a: путь к спорному документу
    :param image_path_b: путь к образцу для сравнения
    :return: текстовый результат анализа сходства
    """
    similarity, details = comparative.compare_images(image_path_a, image_path_b)
    interpretation = comparative.interpret_similarity(similarity)
    report_lines = [
        "Сравнение сходных почерков:",
        f"Коэффициент сходства: {similarity:.2f}",
        f"Вывод: {interpretation}",
        "Детали:",
        f"  Средний размер букв: {details['size_avg'][0]:.1f} vs {details['size_avg'][1]:.1f}",
        f"  Разгон: {details['spacing_avg'][0]:.1f} vs {details['spacing_avg'][1]:.1f}",
        f"  Наклон: {details['slant'][0]:.1f}° vs {details['slant'][1]:.1f}°",
        f"  Связность: {details['connectivity'][0]:.2f} vs {details['connectivity'][1]:.2f}"
    ]
    return '\n'.join(report_lines)