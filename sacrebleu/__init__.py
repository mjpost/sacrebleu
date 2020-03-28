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

__version__ = '1.4.6'
__description__ = 'Hassle-free computation of shareable, comparable, and reproducible BLEU scores'

from .sacrebleu import corpus_bleu, corpus_chrf, sentence_bleu, sentence_chrf, compute_bleu,\
    raw_corpus_bleu, BLEU, CHRF, DATASETS, TOKENIZERS

# more imports for backward compatibility
from .sacrebleu import  ref_stats, bleu_signature, extract_ngrams, extract_char_ngrams, \
    get_corpus_statistics, display_metric, get_sentence_statistics, download_test_set
