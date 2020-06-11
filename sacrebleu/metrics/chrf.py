# -*- coding: utf-8 -*-

# Copyright 2017--2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import re
from collections import Counter
from typing import List, Iterable

from .base import BaseScore, get_signature


class CHRFScore(BaseScore):
    def __init__(self, score, beta, order):
        super().__init__(score)

        self.beta = beta
        # NOTE: order is not in the signature
        self.order = order
        self.prefix = 'chrF{0:d}'.format(self.beta)

    def format(self, width=2, signed=True, short=False, score_only=False):
        if score_only:
            return '{0:.{1}f}'.format(self.score, width)

        prefix = self.prefix
        if signed:
            prefix += '+' + self.signature(short=short)

        return '{pr} = {sc:.{w}f}'.format(pr=prefix, sc=self.score, w=width)


class CHRF:
    # Default values for CHRF
    ORDER = 6

    # default to 2 (per http://www.aclweb.org/anthology/W16-2341)
    BETA = 2

    def __init__(self, args):
        self.name = 'chrf'
        self.include_whitespace = args.chrf_whitespace
        self.order = args.chrf_order
        self.beta = args.chrf_beta

        if self.include_whitespace:
            self._preprocess = lambda x: x
        else:
            self._preprocess = lambda x: re.sub(r'\s+', '', x).strip()

        # Specific signature data
        # 'name': ('short_name', value)
        sig_dict = {
            'numchars': ('n', self.order),
            'space': ('s', str(self.include_whitespace).lower())
        }
        self.__signature = get_signature(args, sig_dict)

    @staticmethod
    def extract_char_ngrams(s: str, n: int) -> Counter:
        """
        Yields counts of character n-grams from string s of order n.
        """
        return Counter([s[i:i + n] for i in range(len(s) - n + 1)])

    @staticmethod
    def compute_chrf(statistics: List[float],
                     order: int,
                     beta: float) -> CHRFScore:

        score = 0.0
        avg_recall = 0.0
        avg_precision = 0.0
        effective_order = 0

        for i in range(order):
            hypotheses_ngrams = statistics[3 * i + 0]
            references_ngrams = statistics[3 * i + 1]
            common_ngrams = statistics[3 * i + 2]
            if hypotheses_ngrams > 0 and references_ngrams > 0:
                avg_precision += common_ngrams / hypotheses_ngrams
                avg_recall += common_ngrams / references_ngrams
                effective_order += 1

        if effective_order == 0:
            avg_precision, avg_recall = 0.0, 0.0
        else:
            avg_precision /= effective_order
            avg_recall /= effective_order

        if avg_precision + avg_recall == 0:
            score = 0.0
        else:
            beta_square = beta ** 2
            score = (1 + beta_square) * (avg_precision * avg_recall)
            score /= ((beta_square * avg_precision) + avg_recall)

        return CHRFScore(score, beta, order)

    def get_sentence_statistics(self, hypothesis: str, reference: str) -> List[float]:
        hypothesis = self._preprocess(hypothesis)
        reference = self._preprocess(reference)
        statistics = [0] * (self.order * 3)
        for i in range(self.order):
            n = i + 1
            hypothesis_ngrams = self.extract_char_ngrams(hypothesis, n)
            reference_ngrams = self.extract_char_ngrams(reference, n)
            common_ngrams = hypothesis_ngrams & reference_ngrams
            statistics[3 * i + 0] = sum(hypothesis_ngrams.values())
            statistics[3 * i + 1] = sum(reference_ngrams.values())
            statistics[3 * i + 2] = sum(common_ngrams.values())
        return statistics

    def get_corpus_statistics(self, hypotheses: Iterable[str],
                              references: Iterable[str]) -> List[float]:
        corpus_statistics = [0] * (self.order * 3)
        for hypothesis, reference in zip(hypotheses, references):
            statistics = self.get_sentence_statistics(hypothesis, reference)
            for i in range(len(statistics)):
                corpus_statistics[i] += statistics[i]
        return corpus_statistics

    def corpus_score(self, hypotheses: Iterable[str], references: Iterable[str]) -> CHRFScore:
        """
        Computes Chrf on a corpus.

        :param hypotheses: Stream of hypotheses.
        :param references: Stream of references
        :return: Chrf score.
        """
        stats = self.get_corpus_statistics(hypotheses, references)
        score = self.compute_chrf(stats, self.order, self.beta)
        score.set_signature(dict(numrefs=('#', 1), **self.__signature))
        return score

    def sentence_score(self, hypothesis: str, reference: str) -> CHRFScore:
        """
        Computes ChrF on a single sentence pair.

        :param hypothesis: Hypothesis string.
        :param reference: Reference string.
        :return: Chrf score.
        """
        stats = self.get_sentence_statistics(hypothesis, reference)
        score = self.compute_chrf(stats, self.order, self.beta)
        score.set_signature(dict(numrefs=('#', 1), **self.__signature))
        return score
