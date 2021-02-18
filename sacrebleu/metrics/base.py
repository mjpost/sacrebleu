
from .. import __version__


class BaseScore:
    """A base score class to derive from."""
    def __init__(self, score):
        self.score = score

    def format(self, width=2, score_only=False, signature=''):
        raise NotImplementedError()

    def __repr__(self):
        return self.format()


class Signature:
    """A convenience class to represent sacreBLEU reproducibility signatures.

    Args:
        args: key-value dictionary passed from the actual metric instance.
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

    def get(self, short: bool = False):
        """Returns a string representation of the signature.

        Args:
            short: If True, shortened signature is produced.
        """
        pairs = []
        for name in sorted(self.info.keys()):
            value = self.info[name]
            if value is not None:
                final_name = self._abbr[name] if short else name
                pairs.append('{}.{}'.format(final_name, value))

        return '+'.join(pairs)

    def __str__(self):
        return self.get()

    def __repr__(self):
        return self.get()
