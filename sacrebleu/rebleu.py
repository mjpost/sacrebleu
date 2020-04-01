#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2020-03-22
from typing import Union, Iterable, List, Set, Dict, Tuple
from itertools import zip_longest
import logging as log
import math
from collections import defaultdict
from pathlib import Path
from .sacrebleu import DEFAULT_TOKENIZER, TOKENIZERS, Result, NGRAM_ORDER, \
    extract_ngrams, ref_stats, VERSION

DEF_SMOOTH_VAL = 1


class NamedResult(Result):
    def __init__(self, name, score):
        self.name = name
        super().__init__(score=score)

    def format(self, width=4) -> str:
        return f'{self.name} {self.score:.{width}f}'

    def signature(self, *args, **kwargs) -> str:
        log.warning('signature() not implemented')
        return ''


class ClassMeasure(NamedResult):

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

    @property
    def f1(self) -> float:
        denr = self.precision + self.recall
        # Note: either zero precision or zero recall leads to zero f1
        return (2 * self.precision * self.recall / denr) if denr > 0 else 0

    def measure(self, measure_name=None):
        measure_name = measure_name or self.measure_name
        return dict(f1=self.f1, precision=self.precision, recall=self.recall)[measure_name]

    def __str__(self):
        return f'ClassMesure[{self.name}, pred/cor/ref={self.preds}/{self.correct}/{self.refs} ' \
            f'P/R/F1={self.precision:g}/{self.recall:g}/{self.f1:g}]'


class MultiClassMeasure(NamedResult):
    """
    Refer to https://datascience.stackexchange.com/a/24051/16531 for micro vs macro
    """

    def __init__(self, name, measures: List[ClassMeasure], average='macro', smooth_value=0):
        self.smooth_value = smooth_value

        def my_log(x):
            assert x > 0, f'{x} > 0 ?'
            return math.log(x)

        avg_types = {'micro': lambda m: smooth_value + m.refs,
                     'micro_sqrt': lambda m: math.sqrt(smooth_value + m.refs),
                     'micro_log': lambda m: my_log(smooth_value + m.refs),
                     'macro': lambda m: 1,
                     }
        assert average in avg_types
        self.measures = measures
        avgs = {}
        for measure_name in ['f1', 'precision', 'recall']:
            wt_scores = [(m.measure(measure_name=measure_name), avg_types[average](m))
                         for m in measures]
            norm = sum(w for score, w in wt_scores)
            avgs[measure_name] = sum(score * w for score, w in wt_scores) / norm
        self.avg_f1 = avgs['f1']
        self.avg_precision = avgs['precision']
        self.avg_recall = avgs['recall']
        self.accuracy = sum(m.correct for m in measures) / sum(m.preds for m in measures)
        super().__init__(name=name, score=self.avg_f1)

    def get_score(self, name):
        return dict(f1=self.avg_f1, precision=self.avg_precision,
                    recall=self.avg_recall, accuracy=self.accuracy)[name]

    def __str__(self):
        return f'MultiClassMesure[{self.name}, ' \
            f'P/R/F1/Acc={self.avg_precision:g}/{self.avg_recall:g}/{self.avg_f1:g}/{self.accuracy}]'


class ReBLEU(NamedResult):
    def __init__(self, name, measures: List[MultiClassMeasure], len_ratio: float, percent=True):
        self.measures = measures
        mean = self._geometric_mean
        self.precision = mean([m.avg_precision for m in measures])
        self.recall = mean([m.avg_recall for m in measures])
        self.f1 = mean([m.avg_f1 for m in measures])
        self.accuracy = mean([m.accuracy for m in measures])

        super().__init__(name=name, score=self.f1)
        self.percent = percent
        self.len_ratio = len_ratio

    @staticmethod
    def _harmonic_mean(scores):
        if any(s == 0 for s in scores):
            score = 0  # if any one of scores is zero => harmonic mean is zero
        else:
            score = len(scores) / sum(1 / s for s in scores)
        return score

    @staticmethod
    def _geometric_mean(scores):
        #  math.exp( sum( map( my_log, precisions[:effective_order])  )  / effective_order)
        if any(s == 0 for s in scores):
            score = 0  # if any one of scores is zero => geometric mean is zero
        else:
            score = math.exp(sum(math.log(score) for score in scores) / len(scores))
        return score

    def format(self, width=4) -> str:
        scaler = 100 if self.percent else 1
        result = f'ReBLEU'
        for name, avg in [('f1', self.f1),
                          ('precision', self.precision),
                          ('recall', self.recall),
                          ('accuracy', self.accuracy)]:
            result += f' {name.title()} {scaler * avg:.{width}f} '
            scores = [scaler * m.get_score(name) for m in self.measures]
            result += '/'.join(f'{score:.{width}f}' for score in scores)

        result += f' ( ratio = {self.len_ratio:.3f} )'
        return result

    def write_report(self, path, args, nrefs):
        if isinstance(path, str):
            path = Path(path)
        scaler, width = 1, 4
        if self.percent:
            scaler, width = 100, 2

        class_stats: Set[ClassMeasure] = set()
        for m in self.measures:
            class_stats.update(m.measures)

        # sort by ascending of ngram, descending of ref_freq, descending of hyp_freq
        class_stats: List[ClassMeasure] = sorted(class_stats, reverse=False,
                                                 key=lambda s: (len(s.name.split()),
                                                                -s.refs, -s.preds))
        delim = '\t'
        ljust = 15

        def format_class_stat(stat: ClassMeasure):
            assert isinstance(stat.name, list)
            row = [" ".join(stat.name).ljust(ljust)]
            row += [str(x) for x in [len(stat.name), stat.refs, stat.preds, stat.correct]]
            row += [f'{x * scaler:.{width}f}' for x in [stat.f1, stat.precision, stat.recall]]
            return delim.join(row)

        with path.open('w', encoding='utf-8', errors='ignore') as out:
            header = ['Type'.ljust(ljust), 'Order', 'Refers', 'Preds', 'Match', 'F1', 'Precisn',
                      'Recall']
            out.write(self.format(width=width) + '\n')
            out.write(self.signature(args, nrefs) + '\n')
            out.write('\n----\n')
            out.write(delim.join(header) + '\n')
            for cs in class_stats:
                out.write(format_class_stat(cs) + '\n')

    def signature(self, args, numrefs):
        """
        Builds a signature that uniquely identifies the scoring parameters used.
        :param args: the arguments passed into the script
        :return: the signature
        """

        # Abbreviations for the signature
        abbr = {
            'test': 't',
            'lang': 'l',
            'smoothval': 'sv',
            'case': 'c',
            'tok': 'tok',
            'numrefs': '#',
            'version': 'v',
            'origlang': 'o',
            'subset': 'S',
            'average': 'a',
            'ngram': 'ng'
        }

        signature = {'tok': args.tokenize,
                     'version': VERSION,
                     'smoothval': args.smooth_value if args.smooth_value is not None else DEF_SMOOTH_VAL,
                     'numrefs': numrefs,
                     'average': args.average,
                     'ngram': args.rebleu_order,
                     'case': 'lc' if args.lc else 'mixed'}

        if args.test_set is not None:
            signature['test'] = args.test_set

        if args.langpair is not None:
            signature['lang'] = args.langpair

        if args.origlang is not None:
            signature['origlang'] = args.origlang
        if args.subset is not None:
            signature['subset'] = args.subset

        sigstr = '+'.join(['{}.{}'.format(abbr[x] if args.short else x, signature[x]) for x in
                           sorted(signature.keys())])
        return sigstr


def _prepare_lines(sys_stream, ref_streams, lowercase, tokenize, force):
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

        if not (force or tokenize == 'none') and lines[0].rstrip().endswith(' .'):
            tokenized_count += 1

            if tokenized_count == 100:
                log.warning("That's 100 lines that end in a tokenized period (' .')")
                log.warning(
                    "It looks like you forgot to detokenize your test data, which may hurt your score.")
                log.warning(
                    "If you insist your data is detokenized, or don't care, you can suppress this message with '--force'.")
        lines = [TOKENIZERS[tokenize](x.rstrip()) for x in lines]
        yield lines


def corpus_rebleu(sys_stream: Union[str, Iterable[str]],
                  ref_streams: Union[str, List[Iterable[str]]],
                  smooth_value=None,
                  force=False,
                  lowercase=False,
                  tokenize=DEFAULT_TOKENIZER,
                  average='micro',
                  max_order=NGRAM_ORDER, report=None) -> ReBLEU:
    """Produces BLEU scores along with its sufficient statistics from a source against one or more references.

    :param sys_stream: The system stream (a sequence of segments)
    :param ref_streams: A list of one or more reference streams (each a sequence of segments)
    :param smooth_value: For 'floor' smoothing, the floor to use
    :param force: Ignore data that looks already tokenized
    :param lowercase: Lowercase the data
    :param tokenize: The tokenizer to use
    :return: a BLEU object containing everything you'd want
    """
    if smooth_value is None:
        smooth_value = DEF_SMOOTH_VAL
    assert smooth_value >= 0

    lines = _prepare_lines(sys_stream, ref_streams, lowercase, tokenize, force)
    gram_stats, ref_len, sys_len = n_gram_performance(lines, max_order)

    gram_measure_groups = defaultdict(list)
    for name, measure in gram_stats.items():
        assert name == measure.name
        measure.name = name.split() # convert to list
        gram_measure_groups[len(measure.name)].append(measure)

    # average measure across multiple classes per group
    group_measures = []
    gram_measure_groups = sorted(gram_measure_groups.items(), key=lambda x: x[0])
    for order, order_measures in gram_measure_groups:
        group_measures.append(MultiClassMeasure(name=f'{order}-gram', measures=order_measures,
                                                average=average, smooth_value=smooth_value))

    # Harmonic mean
    assert len(group_measures) == max_order
    len_ratio = sys_len / ref_len
    harm_mean = ReBLEU(measures=group_measures, name=f'ReBLEU', len_ratio=len_ratio)
    return harm_mean


def n_gram_performance(lines: Iterable[List[str]], max_order: int) \
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
        ref_ngrams, closest_diff, closest_len = ref_stats(output, refs, max_order=max_order)
        sys_len += len(output.split())
        ref_len += closest_len

        sys_ngrams = extract_ngrams(output, max_order=max_order)
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
