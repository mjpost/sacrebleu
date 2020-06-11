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


import math
import logging
from collections import Counter
from itertools import zip_longest
from typing import List, Iterable, Union

from ..tokenizers import TOKENIZERS
from ..utils import my_log
from .. import __version__ as VERSION
from .base import BaseScore


class BLEUScore(BaseScore):
    """A convenience class to represent BLEU scores."""
    def __init__(self, score: float, counts, totals, precisions, bp, sys_len, ref_len):
        super().__init__(score)

        self.prefix = 'BLEU'
        self.bp = bp
        self.counts = counts
        self.totals = totals
        self.sys_len = sys_len
        self.ref_len = ref_len
        self.precisions = precisions
        self.prec_str = "/".join(["{:.1f}".format(p) for p in self.precisions])

    def format(self, width=2, score_only=False, signature=''):
        if score_only:
            return '{0:.{1}f}'.format(self.score, width)

        prefix = self.prefix
        if signature:
            prefix += '+' + signature

        s = '{pr} = {sc:.{w}f} {prec} (BP = {bp:.3f} ratio = {r:.3f} hyp_len = {sl:d} ref_len = {rl:d})'.format(
            pr=prefix,
            sc=self.score,
            w=width,
            prec=self.prec_str,
            bp=self.bp,
            r=self.sys_len / self.ref_len,
            sl=self.sys_len,
            rl=self.ref_len)
        return s


class BLEU:
    NGRAM_ORDER = 4

    SMOOTH_DEFAULTS = {
        'floor': 0.0,
        'add-k': 1,
        'exp': None,    # No value is required
        'none': None,   # No value is required
    }

    # Abbreviations for the signature
    ABBR = {
        'test': 't',
        'lang': 'l',
        'smooth': 's',
        'case': 'c',
        'tok': 'tok',
        'numrefs': '#',
        'version': 'v',
        'origlang': 'o',
        'subset': 'S',
    }

    def __init__(self, args):
        # extract relevant arguments
        self.name = 'bleu'
        self.smooth_method = args.smooth_method
        self.smooth_value = args.smooth_value
        # NOTE: this is an issue here and there
        self.num_refs = args.num_refs
        self.force = args.force
        self.lowercase = args.lc
        self._tokenizer = TOKENIZERS[args.tokenize]()
        self.short = False if 'short' not in args else args.short

        # Base signature
        signature = {
            'tok': self._tokenizer.signature(),
            'smooth': self.smooth_method,
            'numrefs': self.num_refs,
            'case': 'lc' if self.lowercase else 'mixed',
            'version': VERSION,
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
    def extract_ngrams(line, min_order=1, max_order=NGRAM_ORDER) -> Counter:
        """Extracts all the ngrams (min_order <= n <= max_order) from a sequence of tokens.

        :param line: A segment containing a sequence of words.
        :param min_order: Minimum n-gram length (default: 1).
        :param max_order: Maximum n-gram length (default: NGRAM_ORDER).
        :return: a dictionary containing ngrams and counts
        """

        ngrams = Counter()
        tokens = line.split()
        for n in range(min_order, max_order + 1):
            for i in range(0, len(tokens) - n + 1):
                ngram = ' '.join(tokens[i: i + n])
                ngrams[ngram] += 1

        return ngrams

    @staticmethod
    def reference_stats(refs, output_len):
        """Extracts reference statistics for a given segment.

        :param refs: A list of segment tokens.
        :param output_len: Hypothesis length for this segment.
        :return: a tuple of (ngrams, closest_diff, closest_len)
        """

        ngrams = Counter()
        closest_diff = None
        closest_len = None

        for ref in refs:
            tokens = ref.split()
            reflen = len(tokens)
            diff = abs(output_len - reflen)
            if closest_diff is None or diff < closest_diff:
                closest_diff = diff
                closest_len = reflen
            elif diff == closest_diff:
                if reflen < closest_len:
                    closest_len = reflen

            ngrams_ref = BLEU.extract_ngrams(ref)
            for ngram in ngrams_ref.keys():
                ngrams[ngram] = max(ngrams[ngram], ngrams_ref[ngram])

        return ngrams, closest_diff, closest_len

    @staticmethod
    def compute_bleu(correct: List[int],
                     total: List[int],
                     sys_len: int,
                     ref_len: int,
                     smooth_method: str = 'none',
                     smooth_value=None,
                     use_effective_order=False) -> BLEUScore:
        """Computes BLEU score from its sufficient statistics. Adds smoothing.

        Smoothing methods (citing "A Systematic Comparison of Smoothing Techniques for Sentence-Level BLEU",
        Boxing Chen and Colin Cherry, WMT 2014: http://aclweb.org/anthology/W14-3346)

        - exp: NIST smoothing method (Method 3)
        - floor: Method 1
        - add-k: Method 2 (generalizing Lin and Och, 2004)
        - none: do nothing.

        :param correct: List of counts of correct ngrams, 1 <= n <= NGRAM_ORDER
        :param total: List of counts of total ngrams, 1 <= n <= NGRAM_ORDER
        :param sys_len: The cumulative system length
        :param ref_len: The cumulative reference length
        :param smooth: The smoothing method to use
        :param smooth_value: The smoothing value for `floor` and `add-k` methods. `None` falls back to default value.
        :param use_effective_order: If true, use the length of `correct` for the n-gram order instead of NGRAM_ORDER.
        :return: A BLEU object with the score (100-based) and other statistics.
        """
        assert smooth_method in BLEU.SMOOTH_DEFAULTS.keys(), \
            "Unknown smooth_method '{}'".format(smooth_method)

        # Fetch the default value for floor and add-k
        if smooth_value is None:
            smooth_value = BLEU.SMOOTH_DEFAULTS[smooth_method]

        precisions = [0 for x in range(BLEU.NGRAM_ORDER)]

        smooth_mteval = 1.
        effective_order = BLEU.NGRAM_ORDER
        for n in range(1, BLEU.NGRAM_ORDER + 1):
            if smooth_method == 'add-k' and n > 1:
                correct[n-1] += smooth_value
                total[n-1] += smooth_value
            if total[n-1] == 0:
                break

            if use_effective_order:
                effective_order = n

            if correct[n-1] == 0:
                if smooth_method == 'exp':
                    smooth_mteval *= 2
                    precisions[n-1] = 100. / (smooth_mteval * total[n-1])
                elif smooth_method == 'floor':
                    precisions[n-1] = 100. * smooth_value / total[n-1]
            else:
                precisions[n-1] = 100. * correct[n-1] / total[n-1]

        # If the system guesses no i-grams, 1 <= i <= NGRAM_ORDER, the BLEU
        # score is 0 (technically undefined). This is a problem for sentence
        # level BLEU or a corpus of short sentences, where systems will get
        # no credit if sentence lengths fall under the NGRAM_ORDER threshold.
        # This fix scales NGRAM_ORDER to the observed maximum order.
        # It is only available through the API and off by default

        if sys_len < ref_len:
            bp = math.exp(1 - ref_len / sys_len) if sys_len > 0 else 0.0
        else:
            bp = 1.0

        score = bp * math.exp(
            sum(map(my_log, precisions[:effective_order])) / effective_order)

        return BLEUScore(
            score, correct, total, precisions, bp, sys_len, ref_len)

    def sentence_score(self, hypothesis: str,
                       references: List[str],
                       use_effective_order: bool = True) -> BLEUScore:
        """
        Computes BLEU on a single sentence pair.

        Disclaimer: computing BLEU on the sentence level is not its intended use,
        BLEU is a corpus-level metric.

        :param hypothesis: Hypothesis string.
        :param reference: Reference string.
        :param use_effective_order: Account for references that are shorter than the largest n-gram.
        :return: a `BLEUScore` object containing everything you'd want
        """
        return self.corpus_score(hypothesis, references,
                                 use_effective_order=use_effective_order)

    def corpus_score(self, sys_stream: Union[str, Iterable[str]],
                     ref_streams: Union[str, List[Iterable[str]]],
                     use_effective_order: bool = False) -> BLEUScore:
        """Produces BLEU scores along with its sufficient statistics from a source against one or more references.

        :param sys_stream: The system stream (a sequence of segments)
        :param ref_streams: A list of one or more reference streams (each a sequence of segments)
        :param use_effective_order: Account for references that are shorter than the largest n-gram.
        :return: a `BLEUScore` object containing everything you'd want
        """

        # Add some robustness to the input arguments
        # NOTE: Are these correct?
        if isinstance(sys_stream, str):
            sys_stream = [sys_stream]

        if isinstance(ref_streams, str):
            ref_streams = [[ref_streams]]

        sys_len = 0
        ref_len = 0

        correct = [0 for n in range(self.NGRAM_ORDER)]
        total = [0 for n in range(self.NGRAM_ORDER)]

        # look for already-tokenized sentences
        tokenized_count = 0

        fhs = [sys_stream] + ref_streams
        for lines in zip_longest(*fhs):
            if None in lines:
                raise EOFError("Source and reference streams have different lengths!")

            if self.lowercase:
                lines = [x.lower() for x in lines]

            if not (self.force or self._tokenizer.signature() == 'none') and lines[0].rstrip().endswith(' .'):
                tokenized_count += 1

                if tokenized_count == 100:
                    logging.warning('That\'s 100 lines that end in a tokenized period (\'.\')')
                    logging.warning('It looks like you forgot to detokenize your test data, which may hurt your score.')
                    logging.warning('If you insist your data is detokenized, or don\'t care, you can suppress this message with \'--force\'.')

            output, *refs = [self._tokenizer(x.rstrip()) for x in lines]

            output_len = len(output.split())
            ref_ngrams, closest_diff, closest_len = BLEU.reference_stats(refs, output_len)

            sys_len += output_len
            ref_len += closest_len

            sys_ngrams = BLEU.extract_ngrams(output)
            for ngram in sys_ngrams.keys():
                n = len(ngram.split())
                correct[n-1] += min(sys_ngrams[ngram], ref_ngrams.get(ngram, 0))
                total[n-1] += sys_ngrams[ngram]

        return self.compute_bleu(
            correct, total, sys_len, ref_len,
            smooth_method=self.smooth_method, smooth_value=self.smooth_value,
            use_effective_order=use_effective_order)
