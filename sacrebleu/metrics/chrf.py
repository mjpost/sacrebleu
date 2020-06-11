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
from typing import List, Iterable, Tuple

from .base import BaseScore
from .. import __version__ as VERSION


class CHRFScore(BaseScore):
    def __init__(self, score: float, beta, order):
        super().__init__(score)

        # FIXME: beta is included in the prefix but neither beta nor order
        # is in the actual signature in version <= 1.4.10
        self.beta = beta
        self.order = order
        self.prefix = 'chrF{0:d}'.format(self.beta)

    def format(self, width=2, score_only=False, signature=''):
        if score_only:
            return '{0:.{1}f}'.format(self.score, width)

        prefix = self.prefix
        if signature:
            prefix += '+' + signature

        return '{pr} = {sc:.{w}f}'.format(pr=prefix, sc=self.score, w=width)


class CHRF:
    # Default values for CHRF
    ORDER = 6

    # default to 2 (per http://www.aclweb.org/anthology/W16-2341)
    BETA = 2

    # Abbreviations for the signature
    ABBR = {
        'test': 't',
        'lang': 'l',
        'numchars': 'n',
        'space': 's',
        'numrefs': '#',
        'version': 'v',
        'origlang': 'o',
        'subset': 'S',
    }

    def __init__(self, args):
        # extract relevant arguments
        self.name = 'chrf'
        # NOTE: Does chrF support multi refs? The main() function seems to
        # always pass the first ref.
        self.num_refs = args.num_refs

        self.include_whitespace = args.chrf_whitespace
        self.order = args.chrf_order
        self.beta = args.chrf_beta
        self.short = False if 'short' not in args else args.short

        if self.include_whitespace:
            self._preprocess = lambda x: x
        else:
            self._preprocess = lambda x: re.sub(r'\s+', '', x).strip()

        # Base signature
        signature = {
            'version': VERSION,
            'space': self.include_whitespace,
            'numchars': self.order,
            'numrefs': self.num_refs,
        }

        if 'test_set' in args and args.test_set is not None:
            signature['test'] = args.test_set

        if 'langpair' in args and args.langpair is not None:
            signature['lang'] = args.langpair

        if 'origlang' in args and args.origlang is not None:
            signature['origlang'] = args.origlang

        if 'subset' in args and args.subset is not None:
            signature['subset'] = args.subset

        sig_pairs = []
        for name in sorted(signature.keys()):
            key = self.ABBR[name] if self.short else name
            sig_pairs.append('{}.{}'.format(key, signature[name]))

        self.signature = '+'.join(sig_pairs)

    @staticmethod
    def extract_char_ngrams(s: str, n: int) -> Counter:
        """
        Yields counts of character n-grams from string s of order n.
        """
        return Counter([s[i:i + n] for i in range(len(s) - n + 1)])

    @staticmethod
    def avg_precision_and_recall(statistics: List[float], order: int) -> Tuple[float, float]:
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
            return 0.0, 0.0
        avg_precision /= effective_order
        avg_recall /= effective_order
        return avg_precision, avg_recall

    @staticmethod
    def compute_chrf(avg_precision, avg_recall, beta: int = BETA) -> float:
        if avg_precision + avg_recall == 0:
            return 0.0
        beta_square = beta ** 2
        score = (1 + beta_square) * (avg_precision * avg_recall)
        return score / ((beta_square * avg_precision) + avg_recall)

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
        avg_precision, avg_recall = self.avg_precision_and_recall(stats, self.order)
        score = self.compute_chrf(avg_precision, avg_recall, beta=self.beta)
        return CHRFScore(score, self.beta, self.order)

    def sentence_score(self, hypothesis: str, reference: str) -> CHRFScore:
        """
        Computes ChrF on a single sentence pair.

        :param hypothesis: Hypothesis string.
        :param reference: Reference string.
        :return: Chrf score.
        """
        stats = self.get_sentence_statistics(hypothesis, reference)
        avg_precision, avg_recall = self.avg_precision_and_recall(stats, self.order)
        score = self.compute_chrf(avg_precision, avg_recall, beta=self.beta)
        return CHRFScore(score, self.beta, self.order)
