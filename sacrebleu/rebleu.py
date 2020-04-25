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

    def __init__(self, name, measures: List[ClassMeasure], average='macro',
                 smooth_value=0, measure_names=('f1', 'precision', 'recall', 'accuracy'),
                 summary='f1', percent=True):
        self.smooth_value = smooth_value
        assert summary in measure_names
        self.percent = percent

        def my_log(x):
            assert x > 0, f'{x} > 0 ?'
            return math.log(x)

        avg_types = {'micro': lambda m: smooth_value + m.refs,
                     'micro_sqrt': lambda m: math.sqrt(smooth_value + m.refs),
                     'micro_log': lambda m: my_log(smooth_value + m.refs),
                     'macro': lambda m: 1,
                     }

        assert average in avg_types
        weight_func = avg_types[average]
        self.measures = measures
        self.avgs = {}
        for measure_name in measure_names:
            if measure_name == 'accuracy':
                self.avgs['accuracy'] = sum(m.correct for m in measures) \
                                        / sum(m.preds for m in measures)
            else:
                wt_scores = [(m.measure(measure_name=measure_name), weight_func(m))
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
            return sum(s * w for s,w in zip(scores, wts)) / sum(wts)
        else:
            return sum(scores) / len(scores)


class ReBLEU_Old(NamedResult):
    def __init__(self, name, measures: List[MultiClassMeasure], len_ratio: float, percent=True):
        self.measures = measures
        mean = Mean.geometric
        self.precision = mean([m.get_score('precision') for m in measures])
        self.recall = mean([m.get_score('recall') for m in measures])
        self.f1 = mean([m.get_score('f1') for m in measures])
        self.accuracy = mean([m.get_score('accuracy') for m in measures])

        super().__init__(name=name, score=self.f1)
        self.percent = percent
        self.len_ratio = len_ratio

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


class NGramGroup(NamedResult):
    """NGramGroup N-grams based on unigrams """

    def __init__(self, name, max_order):
        self.name = name
        self.groups: List[List[ClassMeasure]] = [[] for _ in range(max_order)]

    def add(self, stat: ClassMeasure):
        assert self.name in stat.name
        self.groups[stat.order() - 1].append(stat)

    def measure(self, measure_name=None) -> float:
        #raise Exception('Error')
        if len(self.groups[0]) != 1:
            log.warning(f"{self.name} expected 1 but found {len(self.groups[0])} unigram types")
        assert len(self.groups[0]) == 1  # exactly one unigram
        groups = [g for g in self.groups if g]  # ignore empty groups
        # Unigram F1
        # unigram_score = groups[0][0].measure('f1')
        meas_names = ['f1'] + ['precision'] * (len(groups) -1)
        # higher grams precision
        g_scores = [[cm.measure(m_name) for cm in g] for m_name, g in zip(meas_names, groups)]
        # arithmetic mean within groups
        intra_means = [Mean.arithmetic(g) for g in g_scores]
        # geometric mean across groups
        cross_mean = Mean.geometric(intra_means)
        return cross_mean

    @property
    def score(self) -> float:
        return self.measure('f1')

    @property
    def refs(self) -> int:
        # unigram ref count
        return self.groups[0][0].refs

    @property
    def head(self) -> ClassMeasure:
        # head of the group wich is unigram
        return self.groups[0][0]


class ReBLEU(NamedResult):
    """
    Refer to https://datascience.stackexchange.com/a/24051/16531 for micro vs macro
    """

    def __init__(self, measures: List[NGramGroup], sys_len, ref_len, name='ReBLEU',
                 average='macro', smooth_value=0, percent=True):
        
        assert sys_len >= 0 
        assert ref_len > 0
        self.smooth_value = smooth_value
        self.percent = percent

        def my_log(x):
            assert x > 0, f'{x} > 0 ?'
            return math.log(x)

        avg_types = {'micro': lambda m: smooth_value + m.refs,
                     'micro_sqrt': lambda m: math.sqrt(smooth_value + m.refs),
                     'micro_log': lambda m: my_log(smooth_value + m.refs),
                     'macro': lambda m: 1,
                     }

        assert average in avg_types
        weight_func = avg_types[average]
        self.measures = measures

        wt_scores = [(m.measure(measure_name='precision'), weight_func(m))
                         for m in measures]
        norm = sum(w for score, w in wt_scores)
        avg_score = sum(score * w for score, w in wt_scores) / norm
        self.brevity_penalty = 1.0
        """
        if sys_len < ref_len:
            self.brevity_penalty = math.exp(1 - ref_len / sys_len) if sys_len > 0 else 0.0           
        """
        self.rebleu = self.brevity_penalty * avg_score * (100 if percent else 1)
        self.sys_len, self.ref_len = sys_len, ref_len
        super().__init__(name=name, score=self.rebleu )

    def __str__(self):
        scaler, width = (100, 2) if self.percent else (1, 4)
        return f'{self.name} {self.score:.{width}f} ( BP = {self.brevity_penalty:.3f} ratio = {self.sys_len/self.ref_len:.3f}' \
                              f' hyp_len = {self.sys_len} ref_len = {self.ref_len} )'
    def format(self, width=2):
        return str(self)


    def write_report(self, path, args, nrefs):
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
            row = [stat.name.ljust(ljust), f"{stat.score * scaler:.{width}f}"]
            head: ClassMeasure = stat.head
            row += [str(x) for x in [head.refs, head.preds, head.correct]]
            row += [f'{x * scaler:.{width}f}' for x in [head.f1, head.precision, head.recall]]
            return delim.join(row)

        with path.open('w', encoding='utf-8', errors='ignore') as out:
            header = ['Type'.ljust(ljust), 'Score', 'Refs', 'Preds', 'Match', 'F1', 'Precisn',
                      'Recall']
            out.write(self.format(width=width) + '\n')
            out.write(self.signature(args, nrefs) + '\n')
            out.write('\n----\n')
            out.write(delim.join(header) + '\n')
            for cs in class_stats:
                out.write(format_class_stat(cs) + '\n')

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
                  max_order=NGRAM_ORDER) -> Union[MultiClassMeasure, ReBLEU]:
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

    """
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
    assert len(group_measures) == max_order
    # ReBLEU
    len_ratio = sys_len / ref_len
    rebleu = ReBLEU(measures=group_measures, name=f'ReBLEU', len_ratio=len_ratio)                                                
    """

    gram_stats = gram_stats.values()
    for gs in gram_stats:
        assert isinstance(gs.name, str)
        gs.name = tuple(gs.name.split())  # convert space separated ngram string to tuple

    unigrams = [gs.name[0] for gs in gram_stats if gs.order() == 1]
    groups = {ug: NGramGroup(name=ug, max_order=max_order) for ug in unigrams}
    for gram_stat in gram_stats:
        for gram in gram_stat.name:
            groups[gram].add(gram_stat)

    groups = list(groups.values())
                              
    """ 
    rebleu = MultiClassMeasure('ReBLEU', measures=groups, average=average,
                               smooth_value=smooth_value, measure_names=['default'], summary='default' )
    """
    rebleu = ReBLEU(measures=groups, average=average, smooth_value=smooth_value, sys_len=sys_len, ref_len=ref_len)

    return rebleu


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
