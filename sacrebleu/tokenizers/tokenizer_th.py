# -*- coding: utf-8 -*-

from .tokenizer_none import NoneTokenizer
from pythainlp.tokenize import word_tokenize, syllable_tokenize


class TokenizerThWord(NoneTokenizer):

    def signature(self):
        return 'th_word'

    def __init__(self, tok_func=word_tokenize):
        self.tok_func = tok_func

    def __call__(self, line):
        """
        :param line: input sentence
        :return: tokenized sentence
        """
        line = line.strip()
        return " ".join(self.tok_func(line))


class TokenizerThSyllable(NoneTokenizer):

    def signature(self):
        return 'th_syllable'

    def __init__(self, tok_func=syllable_tokenize):
        self.tok_func = tok_func

    def __call__(self, line):
        """
        :param line: input sentence
        :return: tokenized sentence
        """
        line = line.strip()
        return " ".join(self.tok_func(line))
