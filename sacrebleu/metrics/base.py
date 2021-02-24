from typing import Any, Sequence

from .. import __version__


class Score:
    """A base score class to derive from."""
    def __init__(self, score):
        self.score = score

    def format(self, width=2, score_only=False, signature=''):
        raise NotImplementedError()

    def __repr__(self):
        return self.format()


class Signature:
    """A convenience class to represent sacreBLEU reproducibility signatures.

    :param args: key-value dictionary passed from the actual metric instance.
    """
    def __init__(self, args: dict):
        # Copy the dictionary
        self.args = dict(args)

        # Global items that are shared across all metrics
        self._abbr = {
            'version': 'v',
            'test': 't',
            'lang': 'l',
            'subset': 'S',
            'origlang': 'o',
            'bootstrap': 'bs',  # enabled with bootstrap resampling
        }

        # Global items that are shared across all metrics
        # None's will be ignored
        self.info = {
            'version': __version__,
            'test': self.args.get('test_set', None),
            'lang': self.args.get('langpair', None),
            'origlang': self.args.get('origlang', None),
            'subset': self.args.get('subset', None),
        }

    def get(self, short: bool = False) -> str:
        """Returns a string representation of the signature.

        :param short: If True, shortened signature is produced.
        :returns: A string representation of the signature.
        """
        pairs = []
        for name in sorted(self.info.keys()):
            value = self.info[name]
            if value is not None:
                final_name = self._abbr[name] if short else name
                pairs.append(f'{final_name}.{value}')

        return '+'.join(pairs)

    def update(self, key: str, value: Any):
        """Add a new item or update an existing one."""
        self.info[key] = value

    def __str__(self):
        return self.get()

    def __repr__(self):
        return self.get()


class Metric:
    """An abstract base class to derive from when creating a metric."""

    def corpus_score(self, hyps: Sequence[str],
                     refs: Sequence[Sequence[str]],
                     n_bootstrap: int = 1) -> Score:
        pass

    def sentence_score(self, hyp: str,
                       refs: Sequence[str]) -> Score:
        if not isinstance(hyp, str):
            raise RuntimeError('The argument `hyp` should be a string.')

        if not isinstance(refs, Sequence):
            raise RuntimeError('The argument `refs` should be a sequence of strings.')

        if not isinstance(refs[0], str):
            raise RuntimeError('Each element of `refs` should be a string.')
