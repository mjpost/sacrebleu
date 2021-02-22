import re
from functools import lru_cache

from .tokenizer_base import BaseTokenizer


class TokenizerChrf(BaseTokenizer):
    """A simple tokenizer for the chrF(++) metric that removes whitespaces.
    """

    def signature(self):
        return 'chrf'

    def __init__(self, lowercase: bool = False,
                 include_whitespace: bool = False):
        self.lowercase = lowercase
        if include_whitespace:
            self.whitespace_re = None
        else:
            self.whitespace_re = re.compile(r'\s+')

    @lru_cache(maxsize=2**18)
    def __call__(self, sent: str) -> str:
        if self.lowercase:
            sent = sent.lower()

        if self.whitespace_re:
            sent = self.whitespace_re.sub('', sent)

        return sent.strip()
