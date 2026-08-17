"""The implementation of various metrics."""

from .bleu import BLEU, BLEUScore   # noqa: F401
from .chrf import CHRF, CHRFScore   # noqa: F401
from .ter import TER, TERScore      # noqa: F401
from .wer import WER, WERScore      # noqa: F401

METRICS = {
    'BLEU': BLEU,
    'CHRF': CHRF,
    'TER': TER,
    'WER': WER,
}
