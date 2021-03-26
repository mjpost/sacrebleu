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

import pytest
import sacrebleu

EPSILON = 1e-4

test_sentence_level_chrf = [
    (
        'Co nás nejvíc trápí, protože lékaři si vybírají, kdo bude žít a kdo zemře.',
        ['Nejvíce smutní jsme z toho, že musíme rozhodovat o tom, kdo bude žít a kdo zemře.'],
        39.14078509,
    ),
    (
        'Nebo prostě nemají vybavení, které by jim pomohlo, uvedli lékaři.',
        ['A někdy nemáme ani potřebný materiál, abychom jim pomohli, popsali lékaři.'],
        31.22557079,
    ),
    (
        'Lapali po dechu, jejich životy skončily dřív, než skutečně začaly.',
        ['Lapali po dechu a pak jejich život skončil - dřív, než skutečně mohl začít, připomněli.'],
        57.15704367,
    ),
]


# hypothesis, reference, expected score
# >= 2.0.0: some orders are not fulfilled in epsilon smoothing (chrF++.py and NLTK)
test_cases = [
    (["abcdefg"], ["hijklmnop"], 0.0),
    (["a"], ["b"], 0.0),
    ([""], ["b"], 0.0),
    ([""], ["ref"], 0.0),
    ([""], ["reference"], 0.0),
    (["aa"], ["ab"], 8.3333),
    (["a", "b"], ["a", "c"], 8.3333),
    (["a"], ["a"], 16.6667),
    (["a b c"], ["a b c"], 50.0),
    (["a b c"], ["abc"], 50.0),
    ([" risk assessment must be made of those who are qualified and expertise in the sector - these are the scientists ."],
     ["risk assessment has to be undertaken by those who are qualified and expert in that area - that is the scientists ."], 63.361730),
    ([" Die    Beziehung zwischen  Obama und Netanjahu ist nicht gerade  freundlich. "],
     ["Das Verhältnis zwischen Obama und Netanyahu ist nicht gerade freundschaftlich."], 64.1302698),
    (["Niemand hat die Absicht, eine Mauer zu errichten"], ["Niemand hat die Absicht, eine Mauer zu errichten"], 100.0),
]

# sacreBLEU < 2.0.0 mode
# hypothesis, reference, expected score
test_cases_effective_order = [
    (["a"], ["a"], 100.0),
    ([""], ["reference"], 0.0),
    (["a b c"], ["a b c"], 100.0),
    (["a b c"], ["abc"], 100.0),
    ([""], ["c"], 0.0),
    (["a", "b"], ["a", "c"], 50.0),
    (["aa"], ["ab"], 25.0),
]

test_cases_keep_whitespace = [
    (
        ["Die Beziehung zwischen Obama und Netanjahu ist nicht gerade freundlich."],
        ["Das Verhältnis zwischen Obama und Netanyahu ist nicht gerade freundschaftlich."],
        67.3481606,
    ),
    (
        ["risk assessment must be made of those who are qualified and expertise in the sector - these are the scientists ."],
        ["risk assessment has to be undertaken by those who are qualified and expert in that area - that is the scientists ."],
        65.2414427,
    ),
]


@pytest.mark.parametrize("hypotheses, references, expected_score", test_cases)
def test_chrf(hypotheses, references, expected_score):
    score = sacrebleu.corpus_chrf(
        hypotheses, [references], char_order=6, word_order=0, beta=3,
        eps_smoothing=True).score
    assert abs(score - expected_score) < EPSILON


@pytest.mark.parametrize("hypotheses, references, expected_score", test_cases_effective_order)
def test_chrf_eff_order(hypotheses, references, expected_score):
    score = sacrebleu.corpus_chrf(
        hypotheses, [references], char_order=6, word_order=0, beta=3,
        eps_smoothing=False).score
    assert abs(score - expected_score) < EPSILON


@pytest.mark.parametrize("hypotheses, references, expected_score", test_cases_keep_whitespace)
def test_chrf_keep_whitespace(hypotheses, references, expected_score):
    score = sacrebleu.corpus_chrf(
        hypotheses, [references], char_order=6, word_order=0, beta=3,
        remove_whitespace=False).score
    assert abs(score - expected_score) < EPSILON


@pytest.mark.parametrize("hypothesis, references, expected_score", test_sentence_level_chrf)
def test_chrf_sentence_level(hypothesis, references, expected_score):
    score = sacrebleu.sentence_chrf(hypothesis, references, eps_smoothing=True).score
    assert abs(score - expected_score) < EPSILON
