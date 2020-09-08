#!/usr/bin/env python

import logging as log
from itertools import zip_longest
from pathlib import Path
from typing import Union, Iterable, List, Dict, Tuple

from . import AVG_TYPES
from .base import NamedResult, ClassMeasure, Mean
from .bleu import BLEU, BLEUSignature
from argparse import Namespace


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
        if isinstance(args, dict):
            args = Namespace(**args)
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
