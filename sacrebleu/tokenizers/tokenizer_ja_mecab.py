# -*- coding: utf-8 -*-

import MeCab

from .tokenizer_none import NoneTokenizer


class TokenizerJaMecab(NoneTokenizer):
    def __init__(self):
        self.tagger = MeCab.Tagger("-Owakati")

        # make sure the dictionary is IPA
        # sacreBLEU is only compatible with 0.996.5 for now
        # Please see: https://github.com/mjpost/sacrebleu/issues/94
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
