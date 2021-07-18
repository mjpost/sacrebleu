import os

from collections import defaultdict

from sacrebleu.metrics import BLEU
from sacrebleu.significance import PairedTest

import pytest


def _read_pickle_file():
    import bz2
    import pickle as pkl
    with bz2.BZ2File('./test/wmt17_en_de_systems.pkl.bz2', 'rb') as f:
        data = pkl.load(f)
    return data


# P-values obtained from Moses' significance script (mean of 3 runs)
# Script: scripts/moses-sigdiff.pl (modified to bootstrap samples = 2000)
MOSES_P_VALS = {
    "newstest2017.C-3MA.4959.en-de": 0.00000,
    "newstest2017.FBK.4870.en-de": 0.01267,
    "newstest2017.KIT.4950.en-de": 0.02233,
    "newstest2017.LMU-nmt-reranked.4934.en-de": 0.04383,
    "newstest2017.LMU-nmt-single.4893.en-de": 0.20783,
    "newstest2017.online-A.0.en-de": 0.00000,
    "newstest2017.online-B.0.en-de": 0.38100,
    "newstest2017.online-F.0.en-de": 0.00000,
    "newstest2017.online-G.0.en-de": 0.00000,
    "newstest2017.PROMT-Rule-based.4735.en-de": 0.00000,
    "newstest2017.RWTH-nmt-ensemble.4921.en-de": 0.01167,
    "newstest2017.SYSTRAN.4847.en-de": 0.20983,
    "newstest2017.TALP-UPC.4834.en-de": 0.00000,
    "newstest2017.uedin-nmt.4722.en-de": 0.00000,
    "newstest2017.xmu.4910.en-de": 0.25483,
}

# Obtained from the multeval toolkit, 10,000 AR trials, (BLEU and TER)
# Code: github.com/mjclark/multeval.git
MULTEVAL_P_VALS = {
    "newstest2017.C-3MA.4959.en-de": (0.0001, 0.0001),
    "newstest2017.FBK.4870.en-de": (0.0218, 0.09569),
    "newstest2017.KIT.4950.en-de": (0.0410, 0.0002),
    "newstest2017.LMU-nmt-reranked.4934.en-de": (0.09029, 0.0001),
    "newstest2017.LMU-nmt-single.4893.en-de": (0.58494, 0.0054),
    "newstest2017.online-A.0.en-de": (0.0001, 0.0001),
    "newstest2017.online-B.0.en-de": (0.94111, 0.82242),
    "newstest2017.online-F.0.en-de": (0.0001, 0.0001),
    "newstest2017.online-G.0.en-de": (0.0001, 0.0001),
    "newstest2017.PROMT-Rule-based.4735.en-de": (0.0001, 0.0001),
    "newstest2017.RWTH-nmt-ensemble.4921.en-de": (0.0207, 0.07539),
    "newstest2017.SYSTRAN.4847.en-de": (0.59914, 0.0001),
    "newstest2017.TALP-UPC.4834.en-de": (0.0001, 0.0001),
    "newstest2017.uedin-nmt.4722.en-de": (0.0001, 0.0001),
    "newstest2017.xmu.4910.en-de": (0.71073, 0.0001),
}


SACREBLEU_BS_P_VALS = defaultdict(float)
SACREBLEU_AR_P_VALS = defaultdict(float)

# Load data from pickled file to not bother with WMT17 downloading
named_systems = _read_pickle_file()
_, refs = named_systems.pop()
metrics = {'BLEU': BLEU(references=refs, tokenize='none')}


#########
# BS test
#########
os.environ['SACREBLEU_SEED'] = str(12345)
bs_scores = PairedTest(
    named_systems, metrics, references=None,
    test_type='bs', n_samples=2000)()[1]

for name, result in zip(bs_scores['System'], bs_scores['BLEU']):
    if result.p_value is not None:
        SACREBLEU_BS_P_VALS[name] += result.p_value


###############################################
# AR test (1 run)
# Test only BLEU as TER will take too much time
###############################################
ar_scores = PairedTest(named_systems, metrics, references=None,
                       test_type='ar', n_samples=10000)()[1]

for name, result in zip(ar_scores['System'], ar_scores['BLEU']):
    if result.p_value is not None:
        SACREBLEU_AR_P_VALS[name] += result.p_value


@pytest.mark.parametrize("name, expected_p_val", MOSES_P_VALS.items())
def test_paired_bootstrap(name, expected_p_val):
    p_val = SACREBLEU_BS_P_VALS[name]
    assert abs(p_val - expected_p_val) < 1e-2


@pytest.mark.parametrize("name, expected_p_vals", MULTEVAL_P_VALS.items())
def test_paired_approximate_randomization(name, expected_p_vals):
    expected_bleu_p_val = expected_p_vals[0]
    p_val = SACREBLEU_AR_P_VALS[name]
    assert abs(p_val - expected_bleu_p_val) < 1e-2
