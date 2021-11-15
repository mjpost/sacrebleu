# -*- coding: utf-8 -*-

import os
import urllib.request

from functools import lru_cache
from ..utils import SACREBLEU_DIR
from .tokenizer_base import BaseTokenizer


class TokenizerSPM(BaseTokenizer):
    def signature(self):
        return 'spm-flores'

    def __init__(self):
        try:
            import sentencepiece as spm
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                '\n\nPlease install the sentencepiece library for SPM tokenization:'
                '\n\n  pip install sentencepiece '
            )
        self.sp = spm.SentencePieceProcessor()

        tokenizer_path = os.path.join(SACREBLEU_DIR, "sacrebleu_tokenizer_spm.model")
        if not os.path.exists(tokenizer_path):
            url = "https://dl.fbaipublicfiles.com/fairseq/models/flores/sacrebleu_tokenizer_spm.model"
            urllib.request.urlretrieve(url, tokenizer_path)
        self.sp.Load(tokenizer_path)

    @lru_cache(maxsize=None)
    def __call__(self, line):
        """Tokenizes all the characters in the input line.

        :param line: a segment to tokenize
        :return: the tokenized line
        """
        return " ".join(self.sp.EncodeAsPieces(line))
