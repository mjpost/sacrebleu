# -*- coding: utf-8 -*-

from .bleu import BLEU, BLEUScore
from .chrf import CHRF, CHRFScore

METRICS = {
    'bleu': BLEU,
    'chrf': CHRF,
}
