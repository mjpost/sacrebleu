#!/usr/bin/env python3

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

__version__ = '2.0.0'
__description__ = 'Hassle-free computation of shareable, comparable, and reproducible BLEU, chrF, and TER scores'


from .metrics import BLEU, CHRF, TER

# Backward compatibility functions for old style API access (<= 1.4.10)
from .compat import corpus_bleu, raw_corpus_bleu, sentence_bleu
from .compat import corpus_chrf, sentence_chrf
from .compat import corpus_ter, sentence_ter

# Other shorthands for backward-compatibility with <= 1.4.10
from .metrics.helpers import extract_word_ngrams as extract_ngrams
from .metrics.helpers import extract_char_ngrams
