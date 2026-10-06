"""
Общие настройки тестов.

Тяжёлые зависимости (torchreid, torch, psycopg2) заменяются заглушками ТОЛЬКО если
не установлены, поэтому тесты запускаются без GPU, весов моделей и PostgreSQL.
"""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _stub(name: str, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module
    return module


try:
    import torchreid.utils  # noqa: F401
except Exception:
    _stub("torchreid")
    _stub("torchreid.utils", FeatureExtractor=object)

try:
    from torch.cuda import is_available  # noqa: F401
except Exception:
    _stub("torch")
    _stub("torch.cuda", is_available=lambda: False)

try:
    import psycopg2  # noqa: F401
    import psycopg2.extras  # noqa: F401
except Exception:
    class _Error(Exception):
        pass

    class _OperationalError(_Error):
        pass

    class _ProgrammingError(_Error):
        pass

    _pg = _stub(
        "psycopg2",
        Error=_Error,
        OperationalError=_OperationalError,
        ProgrammingError=_ProgrammingError,
        connect=lambda **kwargs: None,
    )
    _pg.extras = _stub("psycopg2.extras", execute_batch=lambda *a, **k: None)


DIM = 8


def unit(i: int) -> np.ndarray:
    """Единичный вектор: unit(0) и unit(1) ортогональны (cos = 0), unit(i) с самим собой cos = 1."""
    v = np.zeros(DIM)
    v[i] = 1.0
    return v


@pytest.fixture
def unit_vec():
    return unit
