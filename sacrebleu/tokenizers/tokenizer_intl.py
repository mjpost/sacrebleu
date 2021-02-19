import re
import sys
import unicodedata
import pickle as pkl
from pathlib import Path

from ..utils import SACREBLEU_DIR

from .tokenizer_base import BaseTokenizer


class TokenizerV14International(BaseTokenizer):
    UCD_CACHE_FILE = f'ucd_char_cache_{unicodedata.unidata_version}.pkl'
    UCD_CACHE_PATH = Path(SACREBLEU_DIR) / UCD_CACHE_FILE

    def signature(self):
        return 'intl'

    def __generate_char_cache(self):
        """Caches punctuation and symbol character lists for faster lookup."""
        puncts = []
        symbols = []
        for x in range(sys.maxunicode):
            chrx = chr(x)
            # Around ~800 punctuations
            if unicodedata.category(chrx)[0] == 'P':
                puncts.append(chrx)
            # Around ~8K symbols
            elif unicodedata.category(chrx)[0] == 'S':
                symbols.append(chrx)

        # Join
        self._chars_symbols = ''.join(symbols)
        self._chars_puncts = ''.join(puncts)

        # Dump the cache
        with open(self.UCD_CACHE_PATH, 'wb') as f:
            # Protocol 4 introduced in Python 3.4 and backwards-incompatible
            pkl.dump([self._chars_symbols, self._chars_puncts], f, protocol=4)

    def __load_char_cache(self):
        """Loads the unicode character cache from disk if possible."""
        try:
            with open(self.UCD_CACHE_PATH, 'rb') as f:
                self._chars_symbols, self._chars_puncts = pkl.load(f)
        except Exception:
            # Don't bother with any exceptions, just re-generate and cache
            self.__generate_char_cache()

    def __init__(self):
        # Load the unicode character cache
        self.__load_char_cache()

        self._re = [
            # Separate out punctuations preceeded by a non-digit
            (re.compile(r'([^\d])([' + self._chars_puncts + r'])'), r'\1 \2 '),
            # Separate out punctuations followed by a non-digit
            (re.compile(r'([' + self._chars_puncts + r'])([^\d])'), r' \1 \2'),
            # Separate out symbols
            (re.compile('([' + self._chars_symbols + '])'), r' \1 '),
        ]

    def __call__(self, line):
        r"""Tokenize a string following the official BLEU implementation.

        See https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/mteval-v14.pl#L954-L983
        In our case, the input string is expected to be just one line
        and no HTML entities de-escaping is needed.
        So we just tokenize on punctuation and symbols,
        except when a punctuation is preceded and followed by a digit
        (e.g. a comma/dot as a thousand/decimal separator).

        Note that a number (e.g., a year) followed by a dot at the end of
        sentence is NOT tokenized, i.e. the dot stays with the number because
        `s/(\p{P})(\P{N})/ $1 $2/g` does not match this case (unless we add a
        space after each sentence). However, this error is already in the
        original mteval-v14.pl and we want to be consistent with it.
        The error is not present in the non-international version,
        which uses
        `$norm_text = " $norm_text "` (or `norm = " {} ".format(norm)` in Python).

        :param line: the input string
        :return: the tokenized string
        """

        for (_re, repl) in self._re:
            line = _re.sub(repl, line)
        return line.strip()
