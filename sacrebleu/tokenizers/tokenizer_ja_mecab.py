# -*- coding: utf-8 -*-

from .tokenizer_none import NoneTokenizer

_INSTALL_MSG = "Please install mecab-python3 for evaluating Japanese (pip install mecab-python3)."


class TokenizerJaMecab(NoneTokenizer):
    def __init__(self):
        try:
            import MeCab
        except ImportError:
            raise ImportError(_INSTALL_MSG)

        self.tagger = MeCab.Tagger("-Owakati")

        # make sure the dictionary is IPA.
        d = self.tagger.dictionary_info()
        assert d.size == 392126, \
            "Please make sure to use IPA dictionary for MeCab"
        assert d.next is None

    def __call__(self, line):
        """
        Tokenizes an Japanese input line using MeCab morphological analyzer.

        :param line: a segment to tokenize
        :return: the tokenized line
        """
        line = line.strip()
        sentence = self.tagger.parse(line).strip()
        return sentence

    def signature(self):
        """
        Returns the MeCab parameters.

        :return: signature string
        """
        signature = self.tagger.version() + "-IPA"
        return 'ja-mecab-' + signature
