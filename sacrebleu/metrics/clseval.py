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

# -*- coding: utf-8 -*-

"""
This work is based on:
  Macro-average: Rare Types Are Important Too
    Gowda et al (NAACL 2021)
    TODO: update the link to paper
"""

import copy
import logging as log
from argparse import Namespace
from itertools import zip_longest
from pathlib import Path
from typing import Union, Iterable, List, Dict, Tuple

from .base import BaseScore
from .bleu import BLEU, BLEUSignature

DEF_F_BETA = 1
DEF_AVERAGE = 'macro'
DEF_SMOOTH_VAL = 1


class ClassMeasure(BaseScore):
    __slots__ = 'preds', 'refs', 'correct', 'measure_name', 'name'

    def __init__(self, name, preds=0, refs=0, correct=0):
        self.preds = preds
        self.refs = refs
        self.correct = correct
        self.name = name
        super().__init__(score=self.f1)

    @property
    def precision(self) -> float:
        assert 0 <= self.correct <= self.preds
        # Note: zero correct while zero are predicted is perfect precision
        return (self.correct / self.preds) if self.preds > 0 else 1

    @property
    def recall(self) -> float:
        assert 0 <= self.correct <= self.refs
        # Note: zero correct while zero reference is perfect recall
        return (self.correct / self.refs) if self.refs > 0 else 1

    def f_measure(self, beta: float = 1) -> float:
        denr = beta ** 2 * self.precision + self.recall
        if denr == 0:
            # Note: either zero precision or zero recall leads to zero f1
            return 0
        return (1 + beta ** 2) * self.precision * self.recall / denr

    @property
    def f1(self) -> float:
        return self.f_measure(beta=1)

    def measure(self, measure_name):
        cache = dict(f1=self.f1, precision=self.precision, recall=self.recall)
        if measure_name in cache:
            return cache[measure_name]
        elif measure_name.startswith('f'):
            beta = float(measure_name[1:])
            return self.f_measure(beta=beta)
        else:
            raise Exception(f'Unknown measure name : {measure_name}')

    def __str__(self):
        return f'ClassMeasure[{self.name}, pred/cor/ref={self.preds}/{self.correct}/{self.refs} ' \
               f'P/R/F1={self.precision:g}/{self.recall:g}/{self.f1:g}]'

    def format(self, width=4, score_only=False, signature='') -> str:
        if score_only:
            return f'{self.score:.{width}f}'
        prefix = self.name
        if signature:
            prefix += f'+{signature}'
        return f'{prefix} {self.score:.{width}f}'

AVG_TYPES = {
    'macro': lambda _: 1,
    'micro': lambda f: f,  # term frequency
    #    'micro_sqrt': lambda f: math.sqrt(f),
    #    'micro_log': lambda f: 1 + math.log(f),
    #    'micro_inv': lambda f: 1 / f,  # inverted frequency
}


class MultiClassMeasure(BaseScore):
    """
    Consolidates multiple ClassMeasure s into single measure
    """

    def __init__(self, classes: List[ClassMeasure], average='macro', f_beta=1,
                 smooth_value=0, percent=True, signature='', hyp_len=None, ref_len=None):
        self.smooth_value = smooth_value
        self.signature = signature
        self.classes = classes
        self.percent = percent

        wt_func = AVG_TYPES[average]
        wt_scores = [(m.f_measure(beta=f_beta), wt_func(m.refs + smooth_value))
                     for m in classes]
        f_score = sum(score * w for score, w in wt_scores) / sum(w for score, w in wt_scores)
        scaler = 100 if percent else 1
        f_score *= scaler
        name = 'F{beta:g}'.format(beta=f_beta)
        if average in ('macro', 'micro'):
            name = average.title() + name  # eg MacroF1 MicroF1 MacroF2 ...
        self.name = name
        super().__init__(score=f_score)

        # extra info
        self.hyp_len = hyp_len
        self.ref_len = ref_len
        self.avgs = {}
        for n in ('precision', 'recall'):
            wt_scores = [(m.measure(measure_name=n.lower()), wt_func(m.refs + smooth_value))
                         for m in classes]
            norm = sum(w for score, w in wt_scores)
            self.avgs[n] = scaler * sum(score * w for score, w in wt_scores) / norm

    def __str__(self):
        return self.format()

    def format(self, width=2, score_only=False, signature='') -> str:
        if score_only:
            return '{score:.{width}f}'.format(score=self.score, width=width)
        prefix = self.name
        if signature:
            prefix += '+' + str(signature)
        line = '{prefix} {score:.{width}f}  Precision = {p:.{width}f} Recall = {r:.{width}f}'.format(
            prefix=prefix, score=self.score, width=width, p=self.avgs['precision'],
            r=self.avgs['recall'])

        if self.ref_len:
            line += ' len_ratio = {ratio:.3f} hyp_len = {hyp_len} ref_len = {ref_len}'.format(
                ratio=self.hyp_len / self.ref_len, hyp_len=self.hyp_len, ref_len=self.ref_len)
        return line

    def write_report(self, path):
        if isinstance(path, str):
            path = Path(path)
        scaler, width = 1, 4
        if self.percent:
            scaler, width = 100, 2

        # sort by ascending of ngram, descending of ref_freq, descending of hyp_freq
        class_stats: List[ClassMeasure] = sorted(self.classes, reverse=True,
                                                 key=lambda s: (s.refs, s.preds))
        delim = '\t'
        ljust = 15

        def format_class_stat(class_: ClassMeasure):
            row = [" ".join(class_.name).ljust(ljust), f"{class_.score * scaler:.{width}f}"]
            row += [str(x) for x in [class_.refs, class_.preds, class_.correct]]
            row += [f'{class_.measure(measure_name=x) * scaler:.{width}f}'
                    for x in 'f1 precision recall'.split()]
            return delim.join(row)

        with path.open('w', encoding='utf-8', errors='ignore') as out:
            header = ['Type'.ljust(ljust), 'Score', 'Refs', 'Preds', 'Match', 'F1', 'Precisn',
                      'Recall']
            out.write(self.format(width=width) + '\n')
            out.write(str(self.signature) + '\n')
            out.write('\n----\n')
            out.write(delim.join(header) + '\n')
            for cs in class_stats:
                out.write(format_class_stat(cs) + '\n')


def _prepare_lines(sys_stream, ref_streams, lowercase, tokenizer, force):
    # Add some robustness to the input arguments
    if isinstance(sys_stream, str):
        sys_stream = [sys_stream]
    if isinstance(ref_streams, str):
        ref_streams = [[ref_streams]]
    tokenized_count = 0  # look for already-tokenized sentences
    fhs = [sys_stream] + ref_streams
    for lines in zip_longest(*fhs):
        if None in lines:
            raise EOFError("Source and reference streams have different lengths!")

        if lowercase:
            lines = [x.lower() for x in lines]

        if not (force or tokenizer.signature() == 'none') and lines[0].rstrip().endswith(' .'):
            tokenized_count += 1
            if tokenized_count == 100:
                msg = """"That's 100 lines that end in a tokenized period (' .')
                It looks like you forgot to detokenize your test data, which may hurt your score.
                If you insist your data is detokenized, or don't care, you can suppress this message with '--force'."""
                log.warning(msg)
        lines = [tokenizer(x.rstrip()) for x in lines]
        yield lines


class ClassifierEvalSignature(BLEUSignature):

    def __init__(self, args):
        super().__init__(args)
        # Abbreviations for the signature

        self._abbr.update({
            'average': 'a',
            'ngram': 'ng'})

        self.info.update({
            'average': args.average,
            'ngram': args.max_order,
            'smooth': '{meth}.{val}'.format(val=args.smooth_value, meth=args.smooth_method)
        })
        # smoothing does not apply to macro average
        if args.average == 'macro':
            del self.info['smooth']
        if args.average in ('macro', 'micro'):
            # these are common and hence part of name e.g. MacroF, MicroF
            del self.info['average']


class ClassifierEval(BLEU):
    BETA = DEF_F_BETA

    def __init__(self, args, **override_args):

        if isinstance(args, dict):
            args = Namespace(**args)
        else:
            args = copy.deepcopy(args)

        # overwrite args, they are used by signature
        override_args = override_args or {}
        for key, val in override_args.items():
            setattr(args, key, val)
        args.smooth_value = args.smooth_value or DEF_SMOOTH_VAL

        super().__init__(args)
        self.signature = ClassifierEvalSignature(args)
        self.max_order = args.max_order
        assert self.max_order == 1  # only unigrams are used for now; n > 1 is for future work
        self.smooth_value = args.smooth_value
        self.f_beta = override_args.get('f_beta', args.f_beta)
        assert args.average in AVG_TYPES, f'{args.average} is invalid; use:{list(AVG_TYPES.keys())}'
        self.average = args.average
        self.weight_func = AVG_TYPES[args.average]

    def corpus_score(self, sys_stream: Union[str, Iterable[str]],
                     ref_streams: Union[str, List[Iterable[str]]],
                     use_effective_order: bool = False) \
            -> MultiClassMeasure:
        """Produces Multi-class evaluation score from a source against one or more references.

        :param sys_stream: The system stream (a sequence of segments)
        :param ref_streams: A list of one or more reference streams (each a sequence of segments)
        :return: a MultiClassMeasure object containing everything you'd want
        """
        lines = _prepare_lines(sys_stream, ref_streams, self.lc, self.tokenizer, self.force)
        gram_stats, ref_len, hyp_len = self.n_gram_performance(lines, self.max_order)

        class_measures: List[ClassMeasure] = list(gram_stats.values())
        max_oder = 0
        for class_meas in class_measures:
            assert isinstance(class_meas.name, str)
            class_meas.name = tuple(
                class_meas.name.split())  # convert space separated ngram string to tuple
            max_oder = max(max_oder, len(class_meas.name))
        assert max_oder == 1  # only unigrams for now

        score = MultiClassMeasure(classes=class_measures, f_beta=self.f_beta, percent=True,
                                  average=self.average, smooth_value=self.smooth_value,
                                  signature=str(self.signature), hyp_len=hyp_len, ref_len=ref_len)
        return score

    @classmethod
    def n_gram_performance(cls, lines: Iterable[List[str]], max_order: int) \
            -> Tuple[Dict[str, ClassMeasure], int, int]:
        """
        Gets n-gram performance by comparing system output against 1 or more references
        :param lines:  iterator of [output, ref1 ...] where output and ref1,... are tokenized lines
        :param max_order: maximum n_grams order
        :return: gram_performance, ref_len, sys_len
        """
        assert max_order >= 1
        sys_len, ref_len = 0, 0
        gram_stats = {}
        for output, *refs in lines:
            out_toks = output.split()
            ref_ngrams, closest_diff, closest_len = cls.reference_stats(
                refs=refs, output_len=len(out_toks), max_order=max_order)
            sys_len += len(out_toks)
            ref_len += closest_len

            sys_ngrams = cls.extract_ngrams(output, max_order=max_order)
            for ngram in sys_ngrams.keys():  # n-grams that are recalled by sys
                if ngram not in gram_stats:
                    gram_stats[ngram] = ClassMeasure(name=ngram)
                gram_stats[ngram].preds += sys_ngrams[ngram]
                gram_stats[ngram].correct += min(sys_ngrams[ngram], ref_ngrams.get(ngram, 0))
                gram_stats[ngram].refs += ref_ngrams.get(ngram, 0)

            for ngram in ref_ngrams.keys() - sys_ngrams.keys():  # n-grams that are not recalled by sys
                if ngram not in gram_stats:
                    gram_stats[ngram] = ClassMeasure(name=ngram)
                gram_stats[ngram].refs += ref_ngrams[ngram]
                # .cand and .match are zero by default
        return gram_stats, ref_len, sys_len
