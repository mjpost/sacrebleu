#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2017--2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import re
import sys
import unicodedata
import functools

def tokenize_13a(line):
    """
    Tokenizes an input line using a relatively minimal tokenization that is however equivalent to mteval-v13a, used by WMT.

    :param line: a segment to tokenize
    :return: the tokenized line
    """

    norm = line

    # language-independent part:
    norm = norm.replace('<skipped>', '')
    norm = norm.replace('-\n', '')
    norm = norm.replace('\n', ' ')
    norm = norm.replace('&quot;', '"')
    norm = norm.replace('&amp;', '&')
    norm = norm.replace('&lt;', '<')
    norm = norm.replace('&gt;', '>')

    # language-dependent part (assuming Western languages):
    norm = " {} ".format(norm)
    norm = re.sub(r'([\{-\~\[-\` -\&\(-\+\:-\@\/])', ' \\1 ', norm)
    norm = re.sub(r'([^0-9])([\.,])', '\\1 \\2 ', norm)  # tokenize period and comma unless preceded by a digit
    norm = re.sub(r'([\.,])([^0-9])', ' \\1 \\2', norm)  # tokenize period and comma unless followed by a digit
    norm = re.sub(r'([0-9])(-)', '\\1 \\2 ', norm)  # tokenize dash when preceded by a digit
    norm = re.sub(r'\s+', ' ', norm)  # one space only between words
    norm = re.sub(r'^\s+', '', norm)  # no leading space
    norm = re.sub(r'\s+$', '', norm)  # no trailing space

    return norm


class UnicodeRegex:
    """Ad-hoc hack to recognize all punctuation and symbols.

    without depending on https://pypi.python.org/pypi/regex/."""
    @staticmethod
    def _property_chars(prefix):
        return ''.join(chr(x) for x in range(sys.maxunicode)
                       if unicodedata.category(chr(x)).startswith(prefix))

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def punctuation():
        return UnicodeRegex._property_chars('P')

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def nondigit_punct_re():
        return re.compile(r'([^\d])([' + UnicodeRegex.punctuation() + r'])')

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def punct_nondigit_re():
        return re.compile(r'([' + UnicodeRegex.punctuation() + r'])([^\d])')

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def symbol_re():
        return re.compile('([' + UnicodeRegex._property_chars('S') + '])')


def tokenize_v14_international(string):
    r"""Tokenize a string following the official BLEU implementation.

    See https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/mteval-v14.pl#L954-L983
    In our case, the input string is expected to be just one line
    and no HTML entities de-escaping is needed.
    So we just tokenize on punctuation and symbols,
    except when a punctuation is preceded and followed by a digit
    (e.g. a comma/dot as a thousand/decimal separator).

    Note that a number (e.g., a year) followed by a dot at the end of sentence is NOT tokenized,
    i.e. the dot stays with the number because `s/(\p{P})(\P{N})/ $1 $2/g`
    does not match this case (unless we add a space after each sentence).
    However, this error is already in the original mteval-v14.pl
    and we want to be consistent with it.
    The error is not present in the non-international version,
    which uses `$norm_text = " $norm_text "` (or `norm = " {} ".format(norm)` in Python).

    :param string: the input string
    :return: a list of tokens
    """
    string = UnicodeRegex.nondigit_punct_re().sub(r'\1 \2 ', string)
    string = UnicodeRegex.punct_nondigit_re().sub(r' \1 \2', string)
    string = UnicodeRegex.symbol_re().sub(r' \1 ', string)
    return string.strip()


def tokenize_zh(sentence):
    """MIT License
    Copyright (c) 2017 - Shujian Huang <huangsj@nju.edu.cn>

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    The tokenization of Chinese text in this script contains two steps: separate each Chinese
    characters (by utf-8 encoding); tokenize the non Chinese part (following the mteval script).
    Author: Shujian Huang huangsj@nju.edu.cn

    :param sentence: input sentence
    :return: tokenized sentence
    """

    def is_chinese_char(uchar):
        """
        :param uchar: input char in unicode
        :return: whether the input char is a Chinese character.
        """
        if uchar >= u'\u3400' and uchar <= u'\u4db5':  # CJK Unified Ideographs Extension A, release 3.0
            return True
        elif uchar >= u'\u4e00' and uchar <= u'\u9fa5':  # CJK Unified Ideographs, release 1.1
            return True
        elif uchar >= u'\u9fa6' and uchar <= u'\u9fbb':  # CJK Unified Ideographs, release 4.1
            return True
        elif uchar >= u'\uf900' and uchar <= u'\ufa2d':  # CJK Compatibility Ideographs, release 1.1
            return True
        elif uchar >= u'\ufa30' and uchar <= u'\ufa6a':  # CJK Compatibility Ideographs, release 3.2
            return True
        elif uchar >= u'\ufa70' and uchar <= u'\ufad9':  # CJK Compatibility Ideographs, release 4.1
            return True
        elif uchar >= u'\u20000' and uchar <= u'\u2a6d6':  # CJK Unified Ideographs Extension B, release 3.1
            return True
        elif uchar >= u'\u2f800' and uchar <= u'\u2fa1d':  # CJK Compatibility Supplement, release 3.1
            return True
        elif uchar >= u'\uff00' and uchar <= u'\uffef':  # Full width ASCII, full width of English punctuation, half width Katakana, half wide half width kana, Korean alphabet
            return True
        elif uchar >= u'\u2e80' and uchar <= u'\u2eff':  # CJK Radicals Supplement
            return True
        elif uchar >= u'\u3000' and uchar <= u'\u303f':  # CJK punctuation mark
            return True
        elif uchar >= u'\u31c0' and uchar <= u'\u31ef':  # CJK stroke
            return True
        elif uchar >= u'\u2f00' and uchar <= u'\u2fdf':  # Kangxi Radicals
            return True
        elif uchar >= u'\u2ff0' and uchar <= u'\u2fff':  # Chinese character structure
            return True
        elif uchar >= u'\u3100' and uchar <= u'\u312f':  # Phonetic symbols
            return True
        elif uchar >= u'\u31a0' and uchar <= u'\u31bf':  # Phonetic symbols (Taiwanese and Hakka expansion)
            return True
        elif uchar >= u'\ufe10' and uchar <= u'\ufe1f':
            return True
        elif uchar >= u'\ufe30' and uchar <= u'\ufe4f':
            return True
        elif uchar >= u'\u2600' and uchar <= u'\u26ff':
            return True
        elif uchar >= u'\u2700' and uchar <= u'\u27bf':
            return True
        elif uchar >= u'\u3200' and uchar <= u'\u32ff':
            return True
        elif uchar >= u'\u3300' and uchar <= u'\u33ff':
            return True

        return False

    sentence = sentence.strip()
    sentence_in_chars = ""
    for char in sentence:
        if is_chinese_char(char):
            sentence_in_chars += " "
            sentence_in_chars += char
            sentence_in_chars += " "
        else:
            sentence_in_chars += char
    sentence = sentence_in_chars

    # TODO: the code above could probably be replaced with the following line:
    # import regex
    # sentence = regex.sub(r'(\p{Han})', r' \1 ', sentence)

    # tokenize punctuation
    sentence = re.sub(r'([\{-\~\[-\` -\&\(-\+\:-\@\/])', r' \1 ', sentence)

    # tokenize period and comma unless preceded by a digit
    sentence = re.sub(r'([^0-9])([\.,])', r'\1 \2 ', sentence)

    # tokenize period and comma unless followed by a digit
    sentence = re.sub(r'([\.,])([^0-9])', r' \1 \2', sentence)

    # tokenize dash when preceded by a digit
    sentence = re.sub(r'([0-9])(-)', r'\1 \2 ', sentence)

    # one space only between words
    sentence = re.sub(r'\s+', r' ', sentence)

    # no leading or trailing spaces
    sentence = sentence.strip()

    return sentence


class TokenizeMeCab:
    def __init__(self):
        import MeCab
        self.tagger = MeCab.Tagger("-Owakati")
        # make sure the dictionary is IPA.
        d = self.tagger.dictionary_info()
        assert d.size == 392126, "Please make sure to use IPA dictionary for MeCab"
        assert d.next is None

    def tokenize(self, line):
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

        return signature


TOKENIZERS = {
    '13a': tokenize_13a,
    'intl': tokenize_v14_international,
    'zh': tokenize_zh,
    'ja-mecab': TokenizeMeCab().tokenize,
    'none': lambda x: x,
}