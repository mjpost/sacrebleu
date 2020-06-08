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

from .tokenizer_none import NoneTokenizer


class Tokenizer13a(NoneTokenizer):

    def signature(self):
        return '13a'

    def __call__(self, line):
        """Tokenizes an input line using a relatively minimal tokenization
        that is however equivalent to mteval-v13a, used by WMT.

        :param line: a segment to tokenize
        :return: the tokenized line
        """

        # language-independent part:
        line = line.replace('<skipped>', '')
        line = line.replace('-\n', '')
        line = line.replace('\n', ' ')
        line = line.replace('&quot;', '"')
        line = line.replace('&amp;', '&')
        line = line.replace('&lt;', '<')
        line = line.replace('&gt;', '>')

        # language-dependent part (assuming Western languages):
        line = " {} ".format(line)
        line = re.sub(r'([\{-\~\[-\` -\&\(-\+\:-\@\/])', ' \\1 ', line)

        # tokenize period and comma unless preceded by a digit
        line = re.sub(r'([^0-9])([\.,])', '\\1 \\2 ', line)
        # tokenize period and comma unless followed by a digit
        line = re.sub(r'([\.,])([^0-9])', ' \\1 \\2', line)
        # tokenize dash when preceded by a digit
        line = re.sub(r'([0-9])(-)', '\\1 \\2 ', line)
        # one space only between words
        line = re.sub(r'\s+', ' ', line)
        # no leading space
        line = re.sub(r'^\s+', '', line)
        # no trailing space
        line = re.sub(r'\s+$', '', line)

        return line
