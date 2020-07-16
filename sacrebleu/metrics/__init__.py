# -*- coding: utf-8 -*-

from .bleu import BLEU
from .chrf import CHRF
from .ter import TER

METRICS = {
    'bleu': BLEU,
    'chrf': CHRF,
    'ter': TER,
}
