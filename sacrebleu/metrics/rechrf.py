#!/usr/bin/env python

from . import AVG_TYPES

from .chrf import CHRF, CHRFScore, CHRFSignature
from .rebleu import ClassMeasure, Mean as mean
from typing import List, Iterable, Dict, Union
from itertools import zip_longest
import collections as coll

class ReCHRFSignature(CHRFSignature):
    def __init__(self, args):
        super().__init__(args)

        self._abbr.update({
            'average': 'a',
        })

        self.info.update({
            'average': args.average,
            'smooth': '{meth}.{val}'.format(val=args.smooth_value, meth=args.smooth_method)
        })


class ReCHRFScore(CHRFScore):

    def __init__(self, prefix, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix = prefix

class ReCHRF(CHRF):

    DEF_SMOOTH_VAL = 1

    def __init__(self, args, **override_args):

        # overwrite args, they are used by signature
        override_args = override_args or {}
        for name, val in override_args.items():
            setattr(args, name, val)
        args.smooth_value = args.smooth_value or self.DEF_SMOOTH_VAL

        super().__init__(args)
        self.name = override_args.get('name', 'ReCHRF')  # overwrite
        self.signature = ReCHRFSignature(args)
        self.smooth_value = args.smooth_value
        assert args.smooth_method == 'add-k'

        assert args.average in AVG_TYPES, f'{args.average} is invalid; ' \
                                                      f'Valid: {list(AVG_TYPES.keys())}'
        self.weight_func = AVG_TYPES[args.average]
        self.prefix = f"{self.name}[β={self.beta:g}]"


    def get_sentence_statistics(self, hypothesis: str,
                                references: List[str]) -> List[Dict[str, ClassMeasure]]:
        # NOTE: multi-reference not supported yet
        reference = references[0]

        hypothesis = self._preprocess(hypothesis)
        reference = self._preprocess(reference)
        gram_stats: List[Dict[str, ClassMeasure]] = [dict() for _ in range(self.order)]
        for i in range(self.order):
            n = i + 1
            hypothesis_ngrams = self.extract_char_ngrams(hypothesis, n)
            reference_ngrams = self.extract_char_ngrams(reference, n)

            all_grams = set(hypothesis_ngrams.keys()) | set(reference_ngrams.keys())
            for gram in all_grams:
                preds = hypothesis_ngrams.get(gram, 0)
                refs = reference_ngrams.get(gram, 0)

                if gram not in gram_stats[i]:
                    gram_stats[i][gram] = ClassMeasure(name=gram)
                meas = gram_stats[i][gram]
                meas.preds += preds
                meas.refs += refs
                meas.correct = min(preds, refs)

        return gram_stats

    def compute_chrf(self, statistics: List[Dict[str, ClassMeasure]]) -> CHRFScore:
        group_scores = []
        for group in statistics:
            if len(group) == 0:
                continue

            measures = list(group.values())
            # f measure withing a group
            f_meas = [meas.f_measure(beta=self.beta) for meas in measures]
            weights = [self.weight_func(meas.refs + self.smooth_value) for meas in measures]

            intra_group_mean = mean.arithmetic(f_meas, wts=weights)
            group_scores.append(intra_group_mean)
        score = mean.geometric(group_scores)
        return ReCHRFScore(prefix=self.prefix, score=score, beta=self.beta, order=self.order)

    def sentence_score(self, hypothesis: str, references: List[str]) -> CHRFScore:
        """
        Computes ChrF on a single sentence pair.

        :param hypothesis: Hypothesis string.
        :param references: Reference string(s).
        :return: Chrf score.
        """
        stats = self.get_sentence_statistics(hypothesis, references)
        return self.compute_chrf(stats)


    def corpus_score(self, sys_stream: Union[str, Iterable[str]],
                     ref_streams: Union[str, List[Iterable[str]]]) -> CHRFScore:
        """
        Computes Chrf on a corpus.

        :param hypotheses: Stream of hypotheses.
        :param references: Stream of references.
        :return: Chrf score.
        """

        # Add some robustness to the input arguments
        if isinstance(sys_stream, str):
            sys_stream = [sys_stream]

        if isinstance(ref_streams, str):
            ref_streams = [[ref_streams]]

        corpus_stats = [coll.defaultdict(lambda : [0, 0, 0]) for _ in range(self.order)]

        fhs = [sys_stream] + ref_streams
        for lines in zip_longest(*fhs):
            if None in lines:
                raise EOFError("Source and reference streams have different lengths!")

            # Unpack
            hypothesis, *refs = lines

            sent_stats = self.get_sentence_statistics(hypothesis, refs)
            for i, group in enumerate(sent_stats):
                for meas in group.values():
                    state = corpus_stats[i][meas.name]
                    state[0] += meas.preds
                    state[1] += meas.refs
                    state[2] += meas.correct

        corpus_stats2 = [dict() for _ in range(self.order)]
        for in_group, out_group in zip(corpus_stats, corpus_stats2):
            for name, (preds, refs, correct) in in_group.items():
                assert correct <= preds and correct <= refs
                out_group[name] = ClassMeasure(name, preds=preds, refs=refs,correct=correct)
        return self.compute_chrf(corpus_stats2)
