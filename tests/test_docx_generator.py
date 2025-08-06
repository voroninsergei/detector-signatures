"""
Tests for the DOCX report generator.

These tests ensure that a simple report can be generated without errors
and that the resulting file exists on disk.  Only the high-level API
is tested; content verification is not performed here.
"""
import os
from signature_detector.generator import generate_conclusion


def test_generate_conclusion_creates_file(tmp_path):
    data = {
        'intro': 'Тестовая вступительная часть.',
        'methodology': 'Использованы тестовые методы.',
        'comparative_analysis': 'Результаты сравнения идентичны.',
        'diagnostic': 'Диагностические выводы отсутствуют.',
        'conclusion': 'Заключение: совпадение.',
        'images': [],
        'tables': []
    }
    out_file = tmp_path / 'report.docx'
    generate_conclusion(data, str(out_file))
    assert out_file.exists()
