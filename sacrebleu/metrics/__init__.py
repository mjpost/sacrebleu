# -*- coding: utf-8 -*-

from .bleu import BLEU, BLEUScore
from .chrf import CHRF, CHRFScore
from .rebleu import ReBLEUScorer
from functools import partial

METRICS = {
    'bleu': BLEU,
    'chrf': CHRF,
    'rebleu': ReBLEUScorer,
}

# some useful special cases of rebleu:
# default smooth_method is exp from CLI, here we override it
MacroScorer = partial(ReBLEUScorer, rebleu_avergae='macro', smooth_method='add-k')
MicroScorer = partial(ReBLEUScorer, rebleu_avergae='micro', smooth_method='add-k')
macro_bleu = partial(MacroScorer, rebleu_order=4, name="MacroBLEU")
micro_bleu = partial(MicroScorer, rebleu_order=4, name="MicroBLEU")
macro_f1 = partial(MacroScorer, rebleu_order=1, name="MacroF1")
micro_f1 = partial(MicroScorer, rebleu_order=1, name="MicroF1")


METRICS.update({
    'macrobleu': macro_bleu,
    'microbleu': micro_bleu,
    'macrof1': macro_f1,
    'microf1': micro_f1,
})
