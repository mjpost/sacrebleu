
from .. import __version__
from typing import List
import math


class BaseScore:
    """A base score class to derive from."""

    __slots__ = ('_score',)

    def __init__(self, score):
        self._score = score

    @property
    def score(self) -> float:
        # Child classes are may override and compute it lazily
        return self._score

    def format(self, width=2, score_only=False, signature=''):
        raise NotImplementedError()

    def __repr__(self):
        return self.format()


class Signature:
    """A convenience class to represent sacreBLEU reproducibility signatures.

    :param args: The resulting `Namespace` returned from `parse_args()`.
    Argument-value pairs from command-line would then be directly added
    to the signature.
    """
    def __init__(self, args):
        # Copy the dictionary
        self.args = dict(args.__dict__)
        self.short = self.args.get('short', False)

        self._abbr = {
            'version': 'v',
            'test': 't',
            'lang': 'l',
            'subset': 'S',
            'origlang': 'o',
        }

        self.info = {
            # None's will be ignored
            'version': __version__,
            'test': self.args.get('test_set', None),
            'lang': self.args.get('langpair', None),
            'origlang': self.args.get('origlang', None),
            'subset': self.args.get('subset', None),
        }

    def __str__(self):
        """Returns a formatted signature string."""
        pairs = []
        for name in sorted(self.info.keys()):
            value = self.info[name]
            if value is not None:
                final_name = self._abbr[name] if self.short else name
                pairs.append('{}.{}'.format(final_name, value))

        return '+'.join(pairs)

    def __repr__(self):
        return self.__str__()


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


def safe_log(x):
    assert x > 0, f'{x} > 0 ?'
    return math.log(x)


AVG_TYPES = {
    'macro': lambda _: 1,
    'micro': lambda f: f,  # term frequency
    'micro_sqrt': lambda f: math.sqrt(f),
    'micro_log': lambda f: 1 + safe_log(f),
    'micro_inv': lambda f: 1 / f,  #inverted frequency
    'micro_inv_sqrt': lambda f: 1 / math.sqrt(f),
    'micro_inv_log': lambda f: 1 / (1 + safe_log(f)),
}


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
