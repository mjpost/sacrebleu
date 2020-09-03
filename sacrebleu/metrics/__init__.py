# -*- coding: utf-8 -*-

from .bleu import BLEU, BLEUScore
from .chrf import CHRF, CHRFScore
import math
from functools import partial

METRICS = {
    'bleu': BLEU,
    'chrf': CHRF
}


def safe_log(x):
    assert x > 0, f'{x} > 0 ?'
    return math.log(x)


AVG_TYPES = {
    'macro': lambda _: 1,
    'micro': lambda f: f,  # term frequency
    'micro_sqrt': lambda f: math.sqrt(f),
    'micro_log': lambda f: 1 + safe_log(f),
    'micro_inv': lambda f: 1 / f,  #inverted frequency
    'micro_inv_sqrt': lambda f: 1 / math.sqrt(f),
    'micro_inv_log': lambda f: 1 / (1 + safe_log(f)),
}

from .rebleu import ReBLEUScorer
from .rechrf import ReCHRF

METRICS.update({'rebleu': ReBLEUScorer, 'rechrf': ReCHRF})

# some useful special cases of rebleu:
# default smooth_method is exp from CLI, here we override it
MacroScorer = partial(ReBLEUScorer, average='macro', smooth_method='add-k')
MicroScorer = partial(ReBLEUScorer, average='micro', smooth_method='add-k')

macro_bleu = partial(MacroScorer, rebleu_order=4, name="MacroBLEU")
micro_bleu = partial(MicroScorer, rebleu_order=4, name="MicroBLEU")
macro_f = partial(MacroScorer, rebleu_order=1, name="MacroF")
micro_f = partial(MicroScorer, rebleu_order=1, name="MicroF")

macro_chrf = partial(ReCHRF, name="MacroCHRF", average='macro', smooth_method='add-k')
micro_chrf = partial(ReCHRF, name="MicroCHRF", average='micro', smooth_method='add-k')

METRICS.update({
    'macrobleu': macro_bleu,
    'microbleu': micro_bleu,
    'macrof': macro_f,
    'microf': micro_f,
    'macrochrf': macro_chrf,
    'microchrf': micro_chrf,
})
