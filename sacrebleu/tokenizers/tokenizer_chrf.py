import string
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
        self.include_whitespace = include_whitespace

    @lru_cache(maxsize=2**18)
    def to_words(self, sent: str):
        if self.lowercase:
            sent = sent.lower()

        # This part is from the original chrF++.py implementation
        # https://github.com/m-popovic/chrF
        words = sent.split()
        tokenized = []
        for w in words:
            if len(w) == 1:
                tokenized.append(w)
            else:
                lastChar = w[-1]
                firstChar = w[0]
                if lastChar in string.punctuation:
                    tokenized += [w[:-1], lastChar]
                elif firstChar in string.punctuation:
                    tokenized += [firstChar, w[1:]]
                else:
                    tokenized.append(w)
        return tokenized

    @lru_cache(maxsize=2**18)
    def to_chars(self, sent: str) -> str:
        if self.lowercase:
            sent = sent.lower()

        if not self.include_whitespace:
            sent = ''.join(sent.split())

        return sent
