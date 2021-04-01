import pytest
import sacrebleu

EPSILON = 1e-3


# Example taken from #98
REF = "producţia de zahăr brut se exprimă în zahăr alb;"
SYS = "Producția de zahăr primă va fi exprimată în ceea ce privește zahărul alb;"

test_cases = [
    # change smoothing
    ('exp', None, False, '13a', 8.493),
    ('none', None, False, '13a', 0.0),
    ('floor', None, False, '13a', 4.51688),    # defaults to 0.1
    ('floor', 0.1, False, '13a', 4.51688),
    ('floor', 0.5, False, '13a', 10.10),
    ('add-k', None, False, '13a', 14.882),     # defaults to 1
    ('add-k', 1, False, '13a', 14.882),
    ('add-k', 2, False, '13a', 21.389),
    # change tok
    ('exp', None, False, 'none', 7.347),
    ('exp', None, False, 'intl', 8.493),
    ('exp', None, False, 'char', 40.8759),
    # change case
    ('exp', None, True, 'char', 42.0267),
]


# Example taken from #141
REF_0 = "okay thanks"
SYS_0 = "this is a cat"

test_cases_zero_bleu = [
    ('exp', None, False, '13a', 0.0),
    ('none', None, False, '13a', 0.0),
    ('floor', None, False, '13a', 0.0),    # defaults to 0.1
    ('floor', 0.1, False, '13a', 0.0),
    ('add-k', None, False, '13a', 0.0),     # defaults to 1
    ('add-k', 1, False, '13a', 0.0),
]


@pytest.mark.parametrize("smooth_method, smooth_value, lowercase, tok, expected_score", test_cases)
def test_compat_sentence_bleu(smooth_method, smooth_value, lowercase, tok, expected_score):
    score = sacrebleu.compat.sentence_bleu(
        SYS, [REF], smooth_method=smooth_method, smooth_value=smooth_value,
        tokenize=tok,
        lowercase=lowercase,
        use_effective_order=True)
    assert abs(score.score - expected_score) < EPSILON


@pytest.mark.parametrize("smooth_method, smooth_value, lowercase, tok, expected_score", test_cases)
def test_api_sentence_bleu(smooth_method, smooth_value, lowercase, tok, expected_score):
    metric = sacrebleu.metrics.BLEU(
        lowercase=lowercase, force=False, tokenize=tok,
        smooth_method=smooth_method, smooth_value=smooth_value,
        effective_order=True)
    score = metric.sentence_score(SYS, [REF])

    assert abs(score.score - expected_score) < EPSILON


@pytest.mark.parametrize("smooth_method, smooth_value, lowercase, tok, expected_score", test_cases_zero_bleu)
def test_api_sentence_bleu_zero(smooth_method, smooth_value, lowercase, tok, expected_score):
    metric = sacrebleu.metrics.BLEU(
        lowercase=lowercase, force=False, tokenize=tok,
        smooth_method=smooth_method, smooth_value=smooth_value,
        effective_order=True)
    score = metric.sentence_score(SYS_0, [REF_0])
    assert abs(score.score - expected_score) < EPSILON
