import re

from .tokenizer_base import BaseTokenizer


class TokenizerChrf(BaseTokenizer):
    """A simple tokenizer for the chrF(++) metric that removes whitespaces.
    """

    def signature(self):
        return 'chrf'

    def __init__(self):
        self.whitespace_re = re.compile(r'\s+')

    def __call__(self, sent: str) -> str:
        return self.whitespace_re.sub('', sent).strip()
