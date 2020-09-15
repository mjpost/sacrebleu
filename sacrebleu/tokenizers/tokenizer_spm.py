# -*- coding: utf-8 -*-

import os
import urllib.request

from .tokenizer_none import NoneTokenizer

class TokenizerSPM(NoneTokenizer):
    def signature(self):
        return 'spm'

    def __init__(self):
        try:
            import sentencepiece as spm
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                '\n\nPlease install the sentencepiece library for SPM tokenization:'
                '\n\n  pip install sentencepiece '
            )
        self.sp = spm.SentencePieceProcessor()
        if not os.path.exists('sacrebleu_tokenizer_spm.model'):
            url = "https://dl.fbaipublicfiles.com/fairseq/models/flores/sacrebleu_tokenizer_spm.model"
            urllib.request.urlretrieve(url, 'sacrebleu_tokenizer_spm.model')
        self.sp.Load("sacrebleu_tokenizer_spm.model")

    def __call__(self, line):
        """Tokenizes all the characters in the input line.

        :param line: a segment to tokenize
        :return: the tokenized line
        """
        return " ".join(self.sp.EncodeAsPieces(line))
