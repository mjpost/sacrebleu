import pytest

from sacrebleu.metrics import BLEU, CHRF
from sacrebleu.significance import PairedTest


@pytest.mark.parametrize('seed', ['0', '12345'])
@pytest.mark.parametrize('test_type', ['ar', 'bs'])
def test_paired_test_seed_is_reproducible(monkeypatch, seed, test_type):
    monkeypatch.setenv('SACREBLEU_SEED', seed)
    references = [[
        'the cat sat on the mat',
        'the dog ran across the park',
        'a bird flew into the tree',
        'the car stopped near the bridge',
    ]]
    systems = [
        ('baseline', [
            'a cat is on the mat',
            'the dog ran in the park',
            'a bird flew into a tree',
            'the car stopped by the bridge',
        ]),
        ('candidate', [
            'the cat sat on a mat',
            'a dog ran across the park',
            'the bird flew into the tree',
            'a car stopped near a bridge',
        ]),
    ]
    runs = []
    for _ in range(3):
        test = PairedTest(
            systems, {'BLEU': BLEU(tokenize='none'), 'chrF': CHRF()}, references,
            test_type=test_type, n_samples=100, n_ar_confidence=100,
        )
        signatures, results = test()
        assert all(sig.info['seed'] == seed for sig in signatures.values())
        runs.append({
            metric: [vars(result) for result in values]
            for metric, values in results.items() if metric != 'System'
        })
    assert runs[0] == runs[1] == runs[2]
