# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from collections import namedtuple
import pytest

import sacrebleu

from sacrebleu.metrics import BLEU


EPSILON = 1e-8

Statistics = namedtuple('Statistics', ['common', 'total'])

test_raw_bleu_cases = [
    # This now returns 0.0 score (#141)
    (["this is a test", "another test"], [["ref1", "ref2"]], 0.0),
    (["this is a test"], [["this is a test"]], 1.0),
    (["this is a fest"], [["this is a test"]], 0.223606797749979)]

# test for README example with empty hypothesis strings check
_refs = [
    ['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.'],
    ['The dog had bit the man.', 'No one was surprised.', 'The man had bitten the dog.'],
]

_hyps = [
    'The dog bit the man.',
    "It wasn't surprising.",
    'The man had just bitten him.',
]

test_corpus_bleu_cases = [
    (_hyps, _refs, {}, 48.530827),   # test for default BLEU settings
    (('', '', ''), _refs, {}, 0.0),  # ensure that empty hypotheses are not removed
    (_hyps, _refs, {'tokenize': 'none'}, 49.1919566),
    (_hyps, _refs, {'tokenize': '13a'}, 48.530827),
    (_hyps, _refs, {'tokenize': 'intl'}, 43.91623493),
    (_hyps, _refs, {'smooth_method': 'none'}, 48.530827),
]

test_case_offset = [(["am I am a character sequence"], [["I am a symbol string sequence a a"]], 0.1555722182, 0)]

# statistic structure:
# - common counts
# - total counts
# - hyp_count
# - ref_count

test_case_statistics = [(["am I am a character sequence"], [["I am a symbol string sequence a a"]],
                         Statistics([4, 2, 1, 0], [6, 5, 4, 3]))]

test_case_scoring = [((Statistics([9, 7, 5, 3], [10, 8, 6, 4]), 11, 11), 0.8375922397)]

test_case_effective_order = [(["test"], [["a test"]], 0.3678794411714425),
                             (["a test"], [["a test"]], 1.0),
                             (["a little test"], [["a test"]], 0.03218297948685433)]


# testing that right score is returned for null statistics and different offsets
# format: stat, offset, expected score
test_case_degenerate_stats = [((Statistics([0, 0, 0, 0], [4, 4, 2, 1]), 0, 1), 0.0, 0.0),
                              ((Statistics([0, 0, 0, 0], [10, 11, 12, 0]), 14, 10), 0.0, 0.0),
                              ((Statistics([0, 0, 0, 0], [0, 0, 0, 0]), 0, 0), 0.0, 0.0),
                              ((Statistics([6, 5, 4, 0], [6, 5, 4, 3]), 6, 6), 0.0, 0.0),
                              ((Statistics([0, 0, 0, 0], [0, 0, 0, 0]), 0, 0), 0.1, 0.0),
                              ((Statistics([0, 0, 0, 0], [0, 0, 0, 0]), 1, 5), 0.01, 0.0)]


@pytest.mark.parametrize("hypotheses, references, expected_bleu", test_raw_bleu_cases)
def test_raw_bleu(hypotheses, references, expected_bleu):
    bleu = sacrebleu.raw_corpus_bleu(hypotheses, references, .01).score / 100
    assert abs(bleu - expected_bleu) < EPSILON


@pytest.mark.parametrize("hypotheses, references, kwargs, expected_bleu", test_corpus_bleu_cases)
def test_corpus_bleu(hypotheses, references, kwargs, expected_bleu):
    bleu = sacrebleu.corpus_bleu(hypotheses, references, **kwargs).score
    assert abs(bleu - expected_bleu) < EPSILON


@pytest.mark.parametrize("hypotheses, references, expected_bleu", test_case_effective_order)
def test_effective_order(hypotheses, references, expected_bleu):
    bleu = sacrebleu.raw_corpus_bleu(hypotheses, references, .01).score / 100
    assert abs(bleu - expected_bleu) < EPSILON


@pytest.mark.parametrize("hypothesis, reference, expected_stat", test_case_statistics)
def test_statistics(hypothesis, reference, expected_stat):
    result = sacrebleu.raw_corpus_bleu(hypothesis, reference, .01)
    stat = Statistics(result.counts, result.totals)
    assert stat == expected_stat


@pytest.mark.parametrize("statistics, expected_score", test_case_scoring)
def test_scoring(statistics, expected_score):
    score = BLEU.compute_bleu(statistics[0].common, statistics[0].total, statistics[1], statistics[2]).score / 100
    assert abs(score - expected_score) < EPSILON


@pytest.mark.parametrize("hypothesis, reference, expected_with_offset, expected_without_offset",
                         test_case_offset)
def test_offset(hypothesis, reference, expected_with_offset, expected_without_offset):
    score_without_offset = sacrebleu.raw_corpus_bleu(hypothesis, reference, 0.0).score / 100
    assert abs(expected_without_offset - score_without_offset) < EPSILON

    # let it use BLEU's internal default of 0.1 through passing `None`
    score_with_offset = sacrebleu.raw_corpus_bleu(hypothesis, reference, None).score / 100
    assert abs(expected_with_offset - score_with_offset) < EPSILON

    # let it use BLEU's internal default of 0.1
    score_with_offset = sacrebleu.raw_corpus_bleu(hypothesis, reference).score / 100
    assert abs(expected_with_offset - score_with_offset) < EPSILON


@pytest.mark.parametrize("statistics, offset, expected_score", test_case_degenerate_stats)
def test_degenerate_statistics(statistics, offset, expected_score):
    score = BLEU.compute_bleu(
        statistics[0].common,
        statistics[0].total,
        statistics[1],
        statistics[2],
        smooth_method='floor', smooth_value=offset).score / 100
    assert score == expected_score


test_bleu_max_order = [
    (1, _hyps, _refs, "77.65"),
    (2, _hyps, _refs, "60.50"),
    (3, _hyps, _refs, "53.93"),
    (4, _hyps, _refs, "48.53"),
    (5, _hyps, _refs, "46.14"),
    (6, _hyps, _refs, "43.28"),
]


@pytest.mark.parametrize("order, hyps, refs, expected_bleu", test_bleu_max_order)
def test_max_ngram_order(order, hyps, refs, expected_bleu):
    bleu = BLEU(max_ngram_order=order).corpus_score(hyps, refs)
    assert f"{bleu.score:.2f}" == expected_bleu
