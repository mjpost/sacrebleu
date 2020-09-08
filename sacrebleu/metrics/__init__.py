# -*- coding: utf-8 -*-

from .base import  AVG_TYPES
from .bleu import BLEU, BLEUScore
from .chrf import CHRF, CHRFScore
from .ter import TER, TERScore

METRICS = {
    'bleu': BLEU,
    'chrf': CHRF,
    'ter': TER,
}


from .rebleu import ReBLEUScorer
from .rechrf import ReCHRF
from .rebleu2 import ReBLEUScorer as ReBLEUScorer2
from functools import partial


METRICS.update(dict(rebleu=ReBLEUScorer, rechrf=ReCHRF, rebleu2=ReBLEUScorer2))

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
