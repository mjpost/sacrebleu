import json
from argparse import Namespace

import pytest

from sacrebleu.metrics import BLEU, CHRF, TER
from sacrebleu.significance import PairedTest
from sacrebleu.utils import print_results_table


@pytest.mark.parametrize('metric_cls', [BLEU, CHRF, TER])
@pytest.mark.parametrize('test_type', ['ar', 'bs'])
def test_paired_confidence_results_support_json(monkeypatch, capsys, metric_cls, test_type):
    monkeypatch.setenv('SACREBLEU_SEED', '12345')
    references = [['the cat sat on the mat', 'the dog ran across the park']]
    systems = [
        ('baseline', references[0]),
        ('candidate', ['a cat is on the mat', 'the dog ran in the park']),
    ]
    metric = metric_cls()
    signatures, results = PairedTest(
        systems, {'metric': metric}, references,
        test_type=test_type, n_samples=40, n_ar_confidence=40,
    )()
    print_results_table(results, signatures, Namespace(format='json'))
    rows = json.loads(capsys.readouterr().out)
    assert [row['system'] for row in rows] == ['baseline', 'candidate']
    metric_name = next(name for name in results if name != 'System')
    baseline = rows[0][metric_name]
    candidate = rows[1][metric_name]
    assert baseline['score'] == pytest.approx(0.0 if metric_cls is TER else 100.0)
    assert baseline['p_value'] is None
    assert 0 < candidate['score'] < 100
    assert 0 < candidate['p_value'] <= 1
    assert isinstance(candidate['mean'], float)
    assert isinstance(candidate['ci'], float)
