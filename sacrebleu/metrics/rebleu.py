#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2020-03-22
from typing import Union, Iterable, List, Set, Dict, Tuple
from itertools import zip_longest
import logging as log
import math
from pathlib import Path
from . import AVG_TYPES
from .base import BaseScore
from .bleu import BLEU, BLEUSignature




class Mean:

    @staticmethod
    def harmonic(scores):
        if any(s == 0 for s in scores):
            score = 0  # if any one of scores is zero => harmonic mean is zero
        else:
            score = len(scores) / sum(1 / s for s in scores)
        return score

    @staticmethod
    def geometric(scores):
        #  math.exp( sum( map( my_log, precisions[:effective_order])  )  / effective_order)
        if any(s == 0 for s in scores):
            score = 0  # if any one of scores is zero => geometric mean is zero
        else:
            score = math.exp(sum(math.log(score) for score in scores) / len(scores))
        return score

    @staticmethod
    def arithmetic(scores, wts=None):
        if wts:
            assert len(scores) == len(wts)
            return sum(s * w for s, w in zip(scores, wts)) / sum(wts)
        else:
            return sum(scores) / len(scores)


class NamedResult(BaseScore):

    __slots__ = ('name',)

    def __init__(self, name, score):
        self.name = name
        super().__init__(score=score)

    def format(self, width=4) -> str:
        return f'{self.name} {self.score:.{width}f}'


class ClassMeasure(NamedResult):

    __slots__ = 'preds', 'refs', 'correct', 'measure_name'

    def __init__(self, name, preds=0, refs=0, correct=0, measure='f1'):
        self.preds = preds
        self.refs = refs
        self.correct = correct
        assert measure in {'f1', 'precision', 'recall'}
        self.measure_name = measure
        super().__init__(score=self.measure(), name=name)

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

    def measure(self, measure_name=None):
        measure_name = measure_name or self.measure_name
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

    @property
    def order(self):
        """
        Gets the n-gram order
        Use this only for n-gram classes
        """
        assert isinstance(self.name, tuple)
        return len(self.name)


class MultiClassMeasure(NamedResult):
    """
    Refer to https://datascience.stackexchange.com/a/24051/16531 for micro vs macro
    """

    __slots__ =  'smooth_value', 'percent', 'measures', 'avgs'

    def __init__(self, name, measures: List[ClassMeasure], average='macro',
                 smooth_value=0, measure_names=('f1', 'precision', 'recall', 'accuracy'),
                 summary='f1', percent=True):
        self.smooth_value = smooth_value
        assert summary in measure_names
        self.percent = percent

        weight_func = AVG_TYPES[average]
        self.measures = measures
        self.avgs = {}
        for measure_name in measure_names:
            if measure_name == 'accuracy':
                self.avgs['accuracy'] = sum(m.correct for m in measures) \
                                        / sum(m.preds for m in measures)
            else:
                wt_scores = [(m.measure(measure_name=measure_name),
                              weight_func(m.refs + smooth_value))
                             for m in measures]
                norm = sum(w for score, w in wt_scores)
                self.avgs[measure_name] = sum(score * w for score, w in wt_scores) / norm

        super().__init__(name=name, score=self.avgs[summary])

    def get_score(self, name):
        return self.avgs[name]

    def __str__(self):
        scaler, width = (100, 2) if self.percent else (1, 4)
        line = '/'.join(n[:2].title() + f'={v * scaler:.{width}f}' for n, v in self.avgs.items())
        return f'MultiClassMeasure[{self.name}, {line}]'


class NGramGroup(NamedResult):
    """NGramGroup N-grams based on unigrams """
    __slots__ =  ('groups', 'beta')

    def __init__(self, name, max_order, beta):
        super().__init__(name, float('-inf'))
        self.groups: List[List[ClassMeasure]] = [[] for _ in range(max_order)]
        self.beta = beta

    def add(self, stat: ClassMeasure):
        assert self.name in stat.name
        self.groups[stat.order - 1].append(stat)

    def measure(self, measure_name=None) -> float:
        assert not measure_name or measure_name == 'default', 'custom measure not supported; please use default'
        if len(self.groups[0]) != 1:
            log.warning(f"{self.name} expected 1 but found {len(self.groups[0])} unigram types")
        assert len(self.groups[0]) == 1  # exactly one unigram
        groups = [g for g in self.groups if g]  # ignore empty groups
        # Unigram F1
        # unigram_score = groups[0][0].measure('f1')
        meas_names = [f'f{self.beta}'] + ['precision'] * (len(groups) - 1)
        # higher grams precision
        g_scores = [[cm.measure(m_name) for cm in g] for m_name, g in zip(meas_names, groups)]
        # arithmetic mean within groups
        intra_means = [Mean.arithmetic(g) for g in g_scores]
        # geometric mean across groups
        cross_mean = Mean.geometric(intra_means)
        return cross_mean

    @property
    def score(self):
        return self.measure(measure_name='default')

    @property
    def refs(self) -> int:
        # unigram ref count
        return self.groups[0][0].refs

    @property
    def head(self) -> ClassMeasure:
        # head of the group wich is unigram
        return self.groups[0][0]


class ReBLEUScore(NamedResult):
    """
    Refer to https://datascience.stackexchange.com/a/24051/16531 for micro vs macro
    """

    def __init__(self,  measures: List[NGramGroup], weights, sys_len, ref_len, signature,
                 name='ReBLEU', percent=True):

        assert sys_len >= 0
        assert ref_len > 0
        self.percent = percent
        assert measures
        assert len(measures) == len(weights)
        self.measures = measures
        self.weights = weights
        self.signature = signature
        self.beta = measures[0].beta
        wt_scores = [(m.measure(measure_name='default'), w) for m, w in zip(measures, weights)]
        norm = sum(w for score, w in wt_scores)
        avg_score = sum(score * w for score, w in wt_scores) / norm
        self.brevity_penalty = 1.0
        """
        if sys_len < ref_len:
            self.brevity_penalty = math.exp(1 - ref_len / sys_len) if sys_len > 0 else 0.0           
        """
        self.rebleu = self.brevity_penalty * avg_score * (100 if percent else 1)
        self.sys_len, self.ref_len = sys_len, ref_len
        super().__init__(name=name, score=self.rebleu)

    def __str__(self):
        return self.format()

    def format(self, width=2, score_only=False, signature=''):
        if score_only:
            return '{0:.{1}f}'.format(self.score, width)
        signature = signature or str(self.signature)
        name = self.name.strip() + f'[β={self.beta:g}]'
        prefix = "{}+{}".format(name, signature) if signature else name

        s = ('{pr} {sc:.{width}f} ( BP = {bp:.3f} ratio = {r:.3f} hyp_len = {hl:d}'
             ' ref_len = {rl:d})').format(pr=prefix, sc=self.score, width=width,
                                          bp=self.brevity_penalty, r=self.sys_len / self.ref_len,
                                          hl=self.sys_len, rl=self.ref_len)
        return s

    def write_report(self, path):
        if isinstance(path, str):
            path = Path(path)
        scaler, width = 1, 4
        if self.percent:
            scaler, width = 100, 2

        # sort by ascending of ngram, descending of ref_freq, descending of hyp_freq
        class_stats: List[NGramGroup] = sorted(self.measures, reverse=True,
                                               key=lambda s: (s.head.refs, s.head.preds))
        delim = '\t'
        ljust = 15

        def format_class_stat(stat: NGramGroup):
            row = [stat.name.ljust(ljust), f"{stat.measure() * scaler:.{width}f}"]
            head: ClassMeasure = stat.head
            row += [str(x) for x in [head.refs, head.preds, head.correct]]
            row += [f'{head.measure(measure_name=x) * scaler:.{width}f}'
                    for x in 'f1 precision recall'.split()]
            return delim.join(row)

        with path.open('w', encoding='utf-8', errors='ignore') as out:
            header = ['Type'.ljust(ljust), 'Score', 'Refs', 'Preds', 'Match', 'F1', 'Precisn',
                      'Recall']
            out.write(self.format(width=width) + '\n')
            out.write(str() + '\n')
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


class ReBLEUSignature(BLEUSignature):
    def __init__(self, args):
        super().__init__(args)
        # Abbreviations for the signature

        self._abbr.update({
            'average': 'a',
            'ngram': 'ng',
            'beta': 'β',
        })

        self.info.update({
            'average': args.average,
            'ngram': args.rebleu_order,
            'smooth': '{meth}.{val}'.format(val=args.smooth_value, meth=args.smooth_method)
        })


class ReBLEUScorer(BLEU):
    ORDER: int = 4
    BETA: float = 1
    AVERAGE: str = 'macro'
    DEF_SMOOTH_VAL = 1

    def __init__(self, args, **override_args):

        # overwrite args, they are used by signature
        override_args = override_args or {}
        for name, val in override_args.items():
            setattr(args, name, val)
        args.smooth_value = args.smooth_value or self.DEF_SMOOTH_VAL

        super().__init__(args)
        self.name = override_args.get('name', 'ReBLEU')  # overwrite
        self.signature = ReBLEUSignature(args)
        self.max_order = args.rebleu_order
        self.smooth_value = args.smooth_value
        self.beta = override_args.get('rebleu_beta', args.rebleu_beta)

        assert args.average in AVG_TYPES, f'{args.average} is invalid; use:{list(AVG_TYPES.keys())}'
        self.weight_func = AVG_TYPES[args.average]

    def corpus_score(self, sys_stream: Union[str, Iterable[str]],
                     ref_streams: Union[str, List[Iterable[str]]]) \
            -> Union[ReBLEUScore]:
        """Produces ReBLEU scores along with its sufficient statistics from a source against one or more references.

        :param sys_stream: The system stream (a sequence of segments)
        :param ref_streams: A list of one or more reference streams (each a sequence of segments)
        :return: a ReBLEU object containing everything you'd want
        """

        lines = _prepare_lines(sys_stream, ref_streams, self.lc, self.tokenizer, self.force)
        gram_stats, ref_len, sys_len = self.n_gram_performance(lines, self.max_order)

        gram_stats = gram_stats.values()
        for gs in gram_stats:
            assert isinstance(gs.name, str)
            gs.name = tuple(gs.name.split())  # convert space separated ngram string to tuple

        unigrams = [gs.name[0] for gs in gram_stats if gs.order == 1]
        groups: Dict[str, NGramGroup] = {ug: NGramGroup(name=ug, max_order=self.max_order,
                                                        beta=self.beta) for ug in unigrams}
        for gram_stat in gram_stats:
            for gram in gram_stat.name:
                groups[gram].add(gram_stat)

        groups: List[NGramGroup] = list(groups.values())
        weights = [self.weight_func(g.refs + self.smooth_value) for g in groups]
        rebleu = ReBLEUScore(measures=groups, weights=weights, sys_len=sys_len, ref_len=ref_len,
                             name=self.name, signature=self.signature)
        return rebleu

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
