import pytest

import sacrebleu

EPSILON = 1e-3

test_cases = [
    (['aaaa bbbb cccc dddd'], ['aaaa bbbb cccc dddd'], 0),  # perfect match
    (['dddd eeee ffff'], ['aaaa bbbb cccc'], 1),  # no overlap, same length
    ([''], ['a'], 1),  # corner case, empty hypothesis
    (['a'], [''], 1),  # corner case, empty reference
    ([''], [''], 0),  # corner case, both reference and hypothesis empty, we define it as 0.0
    (['the cat sat on the mat'], ['the cat sat on a mat'], 1 / 6),  # single word substitution
    (['a b c d'], ['a b d'], 1 / 3),  # single word insertion in the hypothesis
]


@pytest.mark.parametrize("hypotheses, references, expected_score", test_cases)
def test_wer(hypotheses, references, expected_score):
    metric = sacrebleu.metrics.WER()
    score = metric.corpus_score(hypotheses, [references]).score
    assert abs(score - 100 * expected_score) < EPSILON


def test_wer_case_sensitive():
    metric = sacrebleu.metrics.WER(case_sensitive=True)
    score = metric.corpus_score(['The cat sat.'], [['the cat sat.']]).score
    assert abs(score - 100 * (1 / 3)) < EPSILON


def test_wer_case_insensitive_by_default():
    metric = sacrebleu.metrics.WER()
    score = metric.corpus_score(['The cat sat.'], [['the cat sat.']]).score
    assert score == 0
