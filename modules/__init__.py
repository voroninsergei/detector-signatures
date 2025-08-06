"""Модульный пакет для детектора подписей.

Каждый модуль в пакете реализует отдельную методику анализа, описанную в "особенной части" Орловой. На первом этапе эти модули содержат только заглушки, чтобы обозначить структуру будущего кода.
"""

from . import preprocessing  # noqa: F401
from . import basic_analysis  # noqa: F401
from . import digital  # noqa: F401
from . import time_gap  # noqa: F401
from . import similar_handwriting  # noqa: F401
from . import unusual_conditions  # noqa: F401
from . import intentional_change  # noqa: F401
from . import left_hand  # noqa: F401
from . import print_like  # noqa: F401
from . import imitation  # noqa: F401
from . import personality_diagnosis  # noqa: F401
from . import comparative  # noqa: F401
from . import general_features  # noqa: F401