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

EPSILON = 1e-6

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


test_cases = [
    (["Niemand hat die Absicht, eine Mauer zu errichten"], ["Niemand hat die Absicht, eine Mauer zu errichten"], 1.0),
    (["abcdefg"], ["hijklmnop"], 0.0),
    (["a"], ["a"], 1.0),
    (["a"], ["b"], 0.0),
    ([""], ["reference"], 0.0),
    (["a b c"], ["a b c"], 1.0),
    (["a b c"], ["abc"], 1.0),
    ([""], ["c"], 0.0),
    (["a", "b"], ["a", "c"], 0.5),
    #(["aa"], ["ab"], 0.25), # < 2.0.0, with effective order
    (["aa"], ["ab"], 0.83333),  # >= 2.0.0 compatible with chrF++.py implementation
    ([" Die    Beziehung zwischen  Obama und Netanjahu ist nicht gerade  freundlich. "],
     ["Das Verhältnis zwischen Obama und Netanyahu ist nicht gerade freundschaftlich."], 0.64130269831561459),
    ([" risk assessment must be made of those who are qualified and expertise in the sector - these are the scientists ."],
     ["risk assessment has to be undertaken by those who are qualified and expert in that area - that is the scientists ."], 0.63361730303214769)]


test_cases_keep_whitespace = [
    (
        ["Die Beziehung zwischen Obama und Netanjahu ist nicht gerade freundlich."],
        ["Das Verhältnis zwischen Obama und Netanyahu ist nicht gerade freundschaftlich."],
        0.67348160629772402,
    ),
    (
        ["risk assessment must be made of those who are qualified and expertise in the sector - these are the scientists ."],
        ["risk assessment has to be undertaken by those who are qualified and expert in that area - that is the scientists ."],
        0.652414427449,
    ),
]


@pytest.mark.parametrize("hypotheses, references, expected_score", test_cases)
def test_chrf(hypotheses, references, expected_score):
    score = sacrebleu.corpus_chrf(
        hypotheses, [references], char_order=6, word_order=0, beta=3).score
    assert abs(0.01 * score - expected_score) < EPSILON


@pytest.mark.parametrize("hypotheses, references, expected_score", test_cases_keep_whitespace)
def test_chrf_keep_whitespace(hypotheses, references, expected_score):
    score = sacrebleu.corpus_chrf(
        hypotheses, [references], char_order=6, word_order=0, beta=3,
        remove_whitespace=False).score
    assert abs(0.01 * score - expected_score) < EPSILON


@pytest.mark.parametrize("hypothesis, references, expected_score", test_sentence_level_chrf)
def test_chrf_sentence_level(hypothesis, references, expected_score):
    score = sacrebleu.sentence_chrf(hypothesis, references).score
    assert abs(score - expected_score) < EPSILON
