# -*- coding: utf-8 -*-

from functools import partial

from .bleu import BLEU, BLEUScore
from .chrf import CHRF, CHRFScore
from .clseval import AVG_TYPES, DEF_F_BETA, DEF_AVERAGE, DEF_SMOOTH_VAL
from .clseval import ClassifierEval, MultiClassMeasure
from .ter import TER, TERScore

METRICS = {
    'bleu': BLEU,
    'chrf': CHRF,
    'ter': TER,
    'macrof': partial(ClassifierEval, average='macro', smooth_method='add-k', max_order=1),
    'microf': partial(ClassifierEval, average='micro', smooth_method='add-k', max_order=1)
}
