#!/usr/bin/env python
# this REBLEU is as per buckets described by Jon


import logging as log
from itertools import zip_longest
from pathlib import Path
from typing import Union, Iterable, List, Dict, Tuple

from . import AVG_TYPES
from .base import NamedResult, ClassMeasure, Mean
from .bleu import BLEU, BLEUSignature
from .rebleu import  _prepare_lines
from argparse import Namespace


class ReBLEUScore(NamedResult):
    """
    Refer to https://datascience.stackexchange.com/a/24051/16531 for micro vs macro
    """

    def __init__(self, measures: List[List[ClassMeasure]], sys_len, ref_len, signature,
                 weight_func, name='ReBLEU2', percent=True, measure_name='f1'):

        self.percent = percent
        assert measures
        self.measures = measures
        self.signature = signature
        self.measure_name = measure_name
        self.sys_len, self.ref_len = sys_len, ref_len
        scaler = 100 if self.percent else 1
        stats = [[(weight_func(gm.refs), scaler * gm.measure(measure_name=self.measure_name))
                  for gm in group] for group in measures]

        # stat is [(wt1, score1), (wt2, score2), ...]
        # zip(*stats) => ([wt1, wt2, ...], [score1, score2, ...])
        # dict(zip(['wts', 'scores'], zip(*stats)) => {'wts': [wt1, wt2, ...], 'scores': [score1, score2, ...]}
        # **dict expands dict as key=value arguments

        self.group_scores = [Mean.arithmetic(**dict(zip(['wts', 'scores'], zip(*stat))))
                             for stat in stats]
        score = Mean.geometric(self.group_scores)
        super().__init__(name=name, score=score)

        # compute per-word score
        # all the unigrams
        types = {cls.name: [cls] for cls in measures[0]}
        for group in measures[1:]:
            for cls in group:
                types[cls.name].append(cls)
        scores = []
        weights = []
        self.class_measures = {}
        for name, group in types.items():
            # unigram ref count as input to weight function
            w = weight_func(group[0].refs)
            weights.append(w)
            scaled_scores = [scaler * gm.measure(measure_name=self.measure_name) for gm in group]
            s = Mean.geometric(scaled_scores)
            scores.append(s)
            self.class_measures[name] = (s, group)
        self.score2 = Mean.arithmetic(scores=scores, wts=weights)


    def __str__(self):
        return self.format()

    def format(self, width=2, score_only=False, signature=''):
        if score_only:
            return '{0:.{1}f}'.format(self.score, width)
        signature = signature or str(self.signature)
        name = self.name.strip() + f'[{self.measure_name}]'
        prefix = "{}+{}".format(name, signature) if signature else name
        groups = '/'.join([f'{gs:.{width}g}' for gs in self.group_scores])
        s = ('{pr} {sc:.{width}f} {score2:.{width}f} {groups} ( ratio = {r:.3f} hyp_len = {hl:d} ref_len = {rl:d})'
             ).format(pr=prefix, sc=self.score, width=width, groups=groups, score2=self.score2,
                      r=self.sys_len / self.ref_len, hl=self.sys_len, rl=self.ref_len)
        return s

    def write_report(self, path):
        if isinstance(path, str):
            path = Path(path)
        scaler, width = 1, 4
        if self.percent:
            scaler, width = 100, 2

        stats = sorted(self.class_measures.items(), reverse=True,
                       key=lambda x: max(x[1][1][0].refs, x[1][1][0].preds))

        delim = '\t'
        ljust = 8

        def format_class_stat(name, score, group:List[ClassMeasure]):
            row = [name.ljust(ljust), f"{score * scaler:.{width}f}"]


            row += [str(x) if i < 3 else f'{scaler * x:.{width}f}' for c in group for i, x in
                    enumerate([c.refs, c.preds, c.correct, c.precision, c.recall, c.f1])]

            return delim.join(row)

        with path.open('w', encoding='utf-8', errors='ignore') as out:
            header = ['Type'.ljust(ljust), 'Score']
            order = len(stats[0][1][1])
            header +=  [f'{i+1}-{x}' for i in range(order) for x in ['Refs', 'Preds', 'Match', 'Prec', 'Recal', 'F1']]
            out.write(self.format(width=width) + '\n')
            out.write(str() + '\n')
            out.write('\n----\n')
            out.write(delim.join(header) + '\n')
            for name, (score, group) in stats:
                out.write(format_class_stat(name, score, group) + '\n')


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
        self.name = override_args.get('name', 'ReBLEU2')  # overwrite
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
        groups: List[Dict[str, ClassMeasure]] = [{} for _ in range(self.max_order)]

        for gram_stat in gram_stats:
            assert isinstance(gram_stat.name, str)
            gram = tuple(gram_stat.name.split())
            n = len(gram)
            for gm in set(gram):
                if gm not in groups[n - 1]:
                    groups[n - 1][gm] = ClassMeasure(name=gm)
                clas = groups[n - 1][gm]
                clas.preds += gram_stat.preds
                clas.refs += gram_stat.refs
                clas.correct += gram_stat.correct

        measures = [list(g.values()) for g in groups]

        return ReBLEUScore(measures=measures, sys_len=sys_len, ref_len=ref_len,
                           signature=self.signature, measure_name='f1', name=self.name,
                           weight_func=lambda x: self.weight_func(x + self.smooth_value))

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
