from functools import lru_cache
import regex

from .tokenizer_base import BaseTokenizer


class TokenizerV14International(BaseTokenizer):
    """Tokenizes a string following the official BLEU implementation.

    See https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/mteval-v14.pl#L954-L983

    In our case, the input string is expected to be just one line.
    We just tokenize on punctuation and symbols,
    except when a punctuation is preceded and followed by a digit
    (e.g. a comma/dot as a thousand/decimal separator).
    We also recover escaped forms of some punctuations such as '&apos;', '&gt;'
    as these can appear in MT system outputs (See issue #138)

    Note that a number (e.g., a year) followed by a dot at the end of
    sentence is NOT tokenized, i.e. the dot stays with the number because
    `s/(\p{P})(\P{N})/ $1 $2/g` does not match this case (unless we add a
    space after each sentence). However, this error is already in the
    original mteval-v14.pl and we want to be consistent with it.
    The error is not present in the non-international version,
    which uses `$norm_text = " $norm_text "`.

    :param line: the input string to tokenize.
    :return: The tokenized string.
    """

    def signature(self):
        return 'intl'

    def __init__(self):
        self._re = [
            # Separate out punctuations preceeded by a non-digit
            (regex.compile(r'(\P{N})(\p{P})'), r'\1 \2 '),
            # Separate out punctuations followed by a non-digit
            (regex.compile(r'(\p{P})(\P{N})'), r' \1 \2'),
            # Separate out symbols
            (regex.compile(r'(\p{S})'), r' \1 '),
        ]

    @lru_cache(maxsize=2**18)
    def __call__(self, line: str) -> str:
        if '&' in line:
            line = line.replace('&quot;', '"')
            line = line.replace('&amp;', '&')
            line = line.replace('&lt;', '<')
            line = line.replace('&gt;', '>')
            line = line.replace('&apos;', "'")

        for (_re, repl) in self._re:
            line = _re.sub(repl, line)

        return ' '.join(line.strip().split())
