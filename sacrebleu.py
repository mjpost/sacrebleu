#!/usr/bin/env python3
"""
SacréBLEU provides hassle-free computation of shareable, comparable, and reproducible BLEU scores.
Inspired by Rico Sennrich's `multi-bleu-detok.perl`, it produces the official WMT scores but works with plain text.
It also knows all the standard test sets and handles downloading, processing, and tokenization for you.

Why use this version of BLEU?
- It automatically downloads common WMT test sets and processes them to plain text
- It produces a short version string that facilitates cross-paper comparisons
- It properly computes scores on detokenized outputs, using WMT ([Conference on Machine Translation](http://statmt.org/wmt17)) standard tokenization
- It produces the same values as official script (`mteval-v13a.pl`) used by WMT
- It outputs the BLEU score without the comma, so you don't have to remove it with `sed` (Looking at you, `multi-bleu.perl`)

# QUICK START

Install the Python module (Python 3 only)

    pip3 install sacrebleu

This installs a shell script, `sacrebleu`.
(You can also directly run the shell script `sacrebleu.py` in the source repository).

Get a list of available test sets:

    sacrebleu

Download the source for one of the pre-defined test sets:

    sacrebleu -t wmt14 -l de-en --echo src > wmt14-de-en.src

(you can also use long parameter names for readability):

    sacrebleu --test-set wmt14 --langpair de-en --echo src > wmt14-de-en.src

After tokenizing, translating, and detokenizing it, you can score your decoder output easily:

    cat output.detok.txt | sacrebleu -t wmt14 -l de-en

SacréBLEU knows about common WMT test sets, but you can also use it to score system outputs with arbitrary references.
It also works in backwards compatible model where you manually specify the reference(s), similar to the format of `multi-bleu.txt`:

    cat output.detok.txt | sacrebleu REF1 [REF2 ...]

Note that the system output and references will all be tokenized internally.

SacréBLEU generates version strings like the following.
Put them in a footnote in your paper!
Use `--short` for a shorter hash if you like.

    BLEU+case.mixed+lang.de-en+test.wmt17 = 32.97 66.1/40.2/26.6/18.1 (BP = 0.980 ratio = 0.980 hyp_len = 63134 ref_len = 64399)

# MOTIVATION

Comparing BLEU scores is harder than it should be.
Every decoder has its own implementation, often borrowed from Moses, but maybe with subtle changes.
Moses itself has a number of implementations as standalone scripts, with little indication of how they differ (note: they mostly don't, but `multi-bleu.pl` expects tokenized input).
Different flags passed to each of these scripts can produce wide swings in the final score.
All of these may handle tokenization in different ways.
On top of this, downloading and managing test sets is a moderate annoyance.
Sacré bleu!
What a mess.

SacréBLEU aims to solve these problems by wrapping the original Papineni reference implementation together with other useful features.
The defaults are set the way that BLEU should be computed, and furthermore, the script outputs a short version string that allows others to know exactly what you did.
As an added bonus, it automatically downloads and manages test sets for you, so that you can simply tell it to score against 'wmt14', without having to hunt down a path on your local file system.
It is all designed to take BLEU a little more seriously.
After all, even with all its problems, BLEU is the default and---admit it---well-loved metric of our entire research community.
Sacré BLEU.

# VERSION HISTORY

- 1.0.4 (in progress).
   - Small bugfixes, windows compatibility (H/T Christian Federmann)

- 1.0.3 (4 November 2017).
   - Contributions from Christian Federmann:
   - Added explicit support for encoding  
   - Fixed Windows support
   - Bugfix in handling reference length with multiple refs

- version 1.0.1 (1 November 2017).
   - Small bugfix affecting some versions of Python.
   - Code reformatting due to Ozan Çağlayan.

- version 1.0 (23 October 2017).
   - Support for WMT 2008--2017.
   - Single tokenization (v13a) with lowercase fix (proper lower() instead of just A-Z).
   - Chinese tokenization.
   - Tested to match all WMT17 scores on all arcs.

# LICENSE

SacréBLEU is licensed under the Apache 2.0 License.

# CREDITS

This was all Rico Sennrich's idea.
Originally written by Matt Post.
The official version can be found at github.com/mjpost/sacreBLEU
"""

import re
import os
import sys
import math
import gzip
import tarfile
import logging
import urllib.request
import urllib.parse
import argparse

from collections import defaultdict, namedtuple

try:
    # SIGPIPE is not available on Windows machines, throwing an exception.
    from signal import SIGPIPE

    # If SIGPIPE is available, change behaviour to default instead of ignore.
    from signal import signal, SIG_DFL
    signal(SIGPIPE, SIG_DFL)

except ImportError:
    logging.warn('Could not import signal.SIGPIPE (this is expected on Windows machines)')

VERSION = '1.0.4'

# Where to store downloaded test sets.
# Define the environment variable $SACREBLEU, or use the default of ~/.sacrebleu.
#
# Querying for a HOME environment variable can result in None (e.g., on Windows)
# in which case the os.path.join() throws a TypeError. Using expanduser() is
# a safe way to get the user's home folder.
from os.path import expanduser
USERHOME = expanduser("~")
SACREBLEU = os.environ.get('SACREBLEU', os.path.join(USERHOME, '.sacrebleu'))

# This defines data locations.
# At the top level are test sets.
# Beneath each test set, we define the location to download the test data.
# The other keys are each language pair contained in the tarball, and the respective locations of the source and reference data within each.
# Many of these are *.sgm files, which are processed to produced plain text that can be used by this script.
# The canonical location of unpacked, processed data is $SACREBLEU/$TEST/$SOURCE-$TARGET.{$SOURCE,$TARGET}
data = {
    'wmt17': {
        'data': 'http://data.statmt.org/wmt17/translation-task/test.tgz',
        'description': 'Official evaluation data.',
        'cs-en': ['test/newstest2017-csen-src.cs.sgm', 'test/newstest2017-csen-ref.en.sgm'],
        'de-en': ['test/newstest2017-deen-src.de.sgm', 'test/newstest2017-deen-ref.en.sgm'],
        'en-cs': ['test/newstest2017-encs-src.en.sgm', 'test/newstest2017-encs-ref.cs.sgm'],
        'en-de': ['test/newstest2017-ende-src.en.sgm', 'test/newstest2017-ende-ref.de.sgm'],
        'en-fi': ['test/newstest2017-enfi-src.en.sgm', 'test/newstest2017-enfi-ref.fi.sgm'],
        'en-lv': ['test/newstest2017-enlv-src.en.sgm', 'test/newstest2017-enlv-ref.lv.sgm'],
        'en-ru': ['test/newstest2017-enru-src.en.sgm', 'test/newstest2017-enru-ref.ru.sgm'],
        'en-tr': ['test/newstest2017-entr-src.en.sgm', 'test/newstest2017-entr-ref.tr.sgm'],
        'en-zh': ['test/newstest2017-enzh-src.en.sgm', 'test/newstest2017-enzh-ref.zh.sgm'],
        'fi-en': ['test/newstest2017-fien-src.fi.sgm', 'test/newstest2017-fien-ref.en.sgm'],
        'lv-en': ['test/newstest2017-lven-src.lv.sgm', 'test/newstest2017-lven-ref.en.sgm'],
        'ru-en': ['test/newstest2017-ruen-src.ru.sgm', 'test/newstest2017-ruen-ref.en.sgm'],
        'tr-en': ['test/newstest2017-tren-src.tr.sgm', 'test/newstest2017-tren-ref.en.sgm'],
        'zh-en': ['test/newstest2017-zhen-src.zh.sgm', 'test/newstest2017-zhen-ref.en.sgm'],
    },
    'wmt17/B': {
        'data': 'http://data.statmt.org/wmt17/translation-task/test.tgz',
        'description': 'Additional reference for EN-FI and FI-EN.',
        'en-fi': ['test/newstestB2017-enfi-src.en.sgm', 'test/newstestB2017-enfi-ref.fi.sgm'],
        'fi-en': ['test/newstestB2017-fien-src.fi.sgm', 'test/newstestB2017-fien-ref.en.sgm'],
    },
    'wmt17/improved': {
        'data': 'http://data.statmt.org/wmt17/translation-task/test-update-1.tgz',
        'description': 'Improved zh-en and en-zh translations.',
        'en-zh': ['newstest2017-enzh-src.en.sgm', 'newstest2017-enzh-ref.zh.sgm'],
        'zh-en': ['newstest2017-zhen-src.zh.sgm', 'newstest2017-zhen-ref.en.sgm'],
    },
    'wmt16': {
        'data': 'http://data.statmt.org/wmt16/translation-task/test.tgz',
        'description': 'Official evaluation data.',
        'cs-en': ['test/newstest2016-csen-src.cs.sgm', 'test/newstest2016-csen-ref.en.sgm'],
        'de-en': ['test/newstest2016-deen-src.de.sgm', 'test/newstest2016-deen-ref.en.sgm'],
        'en-cs': ['test/newstest2016-encs-src.en.sgm', 'test/newstest2016-encs-ref.cs.sgm'],
        'en-de': ['test/newstest2016-ende-src.en.sgm', 'test/newstest2016-ende-ref.de.sgm'],
        'en-fi': ['test/newstest2016-enfi-src.en.sgm', 'test/newstest2016-enfi-ref.fi.sgm'],
        'en-ro': ['test/newstest2016-enro-src.en.sgm', 'test/newstest2016-enro-ref.ro.sgm'],
        'en-ru': ['test/newstest2016-enru-src.en.sgm', 'test/newstest2016-enru-ref.ru.sgm'],
        'en-tr': ['test/newstest2016-entr-src.en.sgm', 'test/newstest2016-entr-ref.tr.sgm'],
        'fi-en': ['test/newstest2016-fien-src.fi.sgm', 'test/newstest2016-fien-ref.en.sgm'],
        'ro-en': ['test/newstest2016-roen-src.ro.sgm', 'test/newstest2016-roen-ref.en.sgm'],
        'ru-en': ['test/newstest2016-ruen-src.ru.sgm', 'test/newstest2016-ruen-ref.en.sgm'],
        'tr-en': ['test/newstest2016-tren-src.tr.sgm', 'test/newstest2016-tren-ref.en.sgm'],
    },
    'wmt16/B': {
        'data': 'http://data.statmt.org/wmt16/translation-task/test.tgz',
        'description': 'Additional reference for EN-FI.',
        'en-fi': ['test/newstestB2016-enfi-src.en.sgm', 'test/newstestB2016-enfi-ref.fi.sgm'],
    },
    'wmt15': {
        'data': 'http://statmt.org/wmt15/test.tgz',
        'description': 'Official evaluation data.',
        'en-fr': ['test/newsdiscusstest2015-enfr-src.en.sgm', 'test/newsdiscusstest2015-enfr-ref.fr.sgm'],
        'fr-en': ['test/newsdiscusstest2015-fren-src.fr.sgm', 'test/newsdiscusstest2015-fren-ref.en.sgm'],
        'cs-en': ['test/newstest2015-csen-src.cs.sgm', 'test/newstest2015-csen-ref.en.sgm'],
        'de-en': ['test/newstest2015-deen-src.de.sgm', 'test/newstest2015-deen-ref.en.sgm'],
        'en-cs': ['test/newstest2015-encs-src.en.sgm', 'test/newstest2015-encs-ref.cs.sgm'],
        'en-de': ['test/newstest2015-ende-src.en.sgm', 'test/newstest2015-ende-ref.de.sgm'],
        'en-fi': ['test/newstest2015-enfi-src.en.sgm', 'test/newstest2015-enfi-ref.fi.sgm'],
        'en-ru': ['test/newstest2015-enru-src.en.sgm', 'test/newstest2015-enru-ref.ru.sgm'],
        'fi-en': ['test/newstest2015-fien-src.fi.sgm', 'test/newstest2015-fien-ref.en.sgm'],
        'ru-en': ['test/newstest2015-ruen-src.ru.sgm', 'test/newstest2015-ruen-ref.en.sgm']
    },
    'wmt14': {
        'data': 'http://statmt.org/wmt14/test-filtered.tgz',
        'description': 'Official evaluation data.',
        'cs-en': ['test/newstest2014-csen-src.cs.sgm', 'test/newstest2014-csen-ref.en.sgm'],
        'en-cs': ['test/newstest2014-csen-src.en.sgm', 'test/newstest2014-csen-ref.cs.sgm'],
        'de-en': ['test/newstest2014-deen-src.de.sgm', 'test/newstest2014-deen-ref.en.sgm'],
        'en-de': ['test/newstest2014-deen-src.en.sgm', 'test/newstest2014-deen-ref.de.sgm'],
        'en-fr': ['test/newstest2014-fren-src.en.sgm', 'test/newstest2014-fren-ref.fr.sgm'],
        'fr-en': ['test/newstest2014-fren-src.fr.sgm', 'test/newstest2014-fren-ref.en.sgm'],
        'en-hi': ['test/newstest2014-hien-src.en.sgm', 'test/newstest2014-hien-ref.hi.sgm'],
        'hi-en': ['test/newstest2014-hien-src.hi.sgm', 'test/newstest2014-hien-ref.en.sgm'],
        'en-ru': ['test/newstest2014-ruen-src.en.sgm', 'test/newstest2014-ruen-ref.ru.sgm'],
        'ru-en': ['test/newstest2014-ruen-src.ru.sgm', 'test/newstest2014-ruen-ref.en.sgm']
    },
    'wmt14/full': {
        'data': 'http://statmt.org/wmt14/test-full.tgz',
        'description': 'Evaluation data released after official evaluation for further research.',
        'cs-en': ['test-full/newstest2014-csen-src.cs.sgm', 'test-full/newstest2014-csen-ref.en.sgm'],
        'en-cs': ['test-full/newstest2014-csen-src.en.sgm', 'test-full/newstest2014-csen-ref.cs.sgm'],
        'de-en': ['test-full/newstest2014-deen-src.de.sgm', 'test-full/newstest2014-deen-ref.en.sgm'],
        'en-de': ['test-full/newstest2014-deen-src.en.sgm', 'test-full/newstest2014-deen-ref.de.sgm'],
        'en-fr': ['test-full/newstest2014-fren-src.en.sgm', 'test-full/newstest2014-fren-ref.fr.sgm'],
        'fr-en': ['test-full/newstest2014-fren-src.fr.sgm', 'test-full/newstest2014-fren-ref.en.sgm'],
        'en-hi': ['test-full/newstest2014-hien-src.en.sgm', 'test-full/newstest2014-hien-ref.hi.sgm'],
        'hi-en': ['test-full/newstest2014-hien-src.hi.sgm', 'test-full/newstest2014-hien-ref.en.sgm'],
        'en-ru': ['test-full/newstest2014-ruen-src.en.sgm', 'test-full/newstest2014-ruen-ref.ru.sgm'],
        'ru-en': ['test-full/newstest2014-ruen-src.ru.sgm', 'test-full/newstest2014-ruen-ref.en.sgm']
    },
    'wmt13': {
        'data': 'http://statmt.org/wmt13/test.tgz',
        'description': 'Official evaluation data.',
        'cs-en': ['test/newstest2013-src.cs.sgm', 'test/newstest2013-src.en.sgm'],
        'en-cs': ['test/newstest2013-src.en.sgm', 'test/newstest2013-src.cs.sgm'],
        'de-en': ['test/newstest2013-src.de.sgm', 'test/newstest2013-src.en.sgm'],
        'en-de': ['test/newstest2013-src.en.sgm', 'test/newstest2013-src.de.sgm'],
        'es-en': ['test/newstest2013-src.es.sgm', 'test/newstest2013-src.en.sgm'],
        'en-es': ['test/newstest2013-src.en.sgm', 'test/newstest2013-src.es.sgm'],
        'fr-en': ['test/newstest2013-src.fr.sgm', 'test/newstest2013-src.en.sgm'],
        'en-fr': ['test/newstest2013-src.en.sgm', 'test/newstest2013-src.fr.sgm'],
        'ru-en': ['test/newstest2013-src.ru.sgm', 'test/newstest2013-src.en.sgm'],
        'en-ru': ['test/newstest2013-src.en.sgm', 'test/newstest2013-src.ru.sgm']
    },
    'wmt12': {
        'data': 'http://statmt.org/wmt12/test.tgz',
        'description': 'Official evaluation data.',
        'cs-en': ['test/newstest2012-src.cs.sgm', 'test/newstest2012-src.en.sgm'],
        'en-cs': ['test/newstest2012-src.en.sgm', 'test/newstest2012-src.cs.sgm'],
        'de-en': ['test/newstest2012-src.de.sgm', 'test/newstest2012-src.en.sgm'],
        'en-de': ['test/newstest2012-src.en.sgm', 'test/newstest2012-src.de.sgm'],
        'es-en': ['test/newstest2012-src.es.sgm', 'test/newstest2012-src.en.sgm'],
        'en-es': ['test/newstest2012-src.en.sgm', 'test/newstest2012-src.es.sgm'],
        'fr-en': ['test/newstest2012-src.fr.sgm', 'test/newstest2012-src.en.sgm'],
        'en-fr': ['test/newstest2012-src.en.sgm', 'test/newstest2012-src.fr.sgm']
    },
    'wmt11': {
        'data': 'http://statmt.org/wmt11/test.tgz',
        'description': 'Official evaluation data.',
        'cs-en': ['newstest2011-src.cs.sgm', 'newstest2011-src.en.sgm'],
        'en-cs': ['newstest2011-src.en.sgm', 'newstest2011-src.cs.sgm'],
        'de-en': ['newstest2011-src.de.sgm', 'newstest2011-src.en.sgm'],
        'en-de': ['newstest2011-src.en.sgm', 'newstest2011-src.de.sgm'],
        'fr-en': ['newstest2011-src.fr.sgm', 'newstest2011-src.en.sgm'],
        'en-fr': ['newstest2011-src.en.sgm', 'newstest2011-src.fr.sgm'],
        'es-en': ['newstest2011-src.es.sgm', 'newstest2011-src.en.sgm'],
        'en-es': ['newstest2011-src.en.sgm', 'newstest2011-src.es.sgm']
    },
    'wmt10': {
        'data': 'http://statmt.org/wmt10/test.tgz',
        'description': 'Official evaluation data.',
        'cs-en': ['test/newstest2010-src.cz.sgm', 'test/newstest2010-src.en.sgm'],
        'en-cs': ['test/newstest2010-src.en.sgm', 'test/newstest2010-src.cz.sgm'],
        'de-en': ['test/newstest2010-src.de.sgm', 'test/newstest2010-src.en.sgm'],
        'en-de': ['test/newstest2010-src.en.sgm', 'test/newstest2010-src.de.sgm'],
        'es-en': ['test/newstest2010-src.es.sgm', 'test/newstest2010-src.en.sgm'],
        'en-es': ['test/newstest2010-src.en.sgm', 'test/newstest2010-src.es.sgm'],
        'fr-en': ['test/newstest2010-src.fr.sgm', 'test/newstest2010-src.en.sgm'],
        'en-fr': ['test/newstest2010-src.en.sgm', 'test/newstest2010-src.fr.sgm']
    },
    'wmt09': {
        'data': 'http://statmt.org/wmt09/test.tgz',
        'description': 'Official evaluation data.',
        'cs-en': ['test/newstest2009-src.cz.sgm', 'test/newstest2009-src.en.sgm'],
        'en-cs': ['test/newstest2009-src.en.sgm', 'test/newstest2009-src.cz.sgm'],
        'de-en': ['test/newstest2009-src.de.sgm', 'test/newstest2009-src.en.sgm'],
        'en-de': ['test/newstest2009-src.en.sgm', 'test/newstest2009-src.de.sgm'],
        'es-en': ['test/newstest2009-src.es.sgm', 'test/newstest2009-src.en.sgm'],
        'en-es': ['test/newstest2009-src.en.sgm', 'test/newstest2009-src.es.sgm'],
        'fr-en': ['test/newstest2009-src.fr.sgm', 'test/newstest2009-src.en.sgm'],
        'en-fr': ['test/newstest2009-src.en.sgm', 'test/newstest2009-src.fr.sgm'],
        'hu-en': ['test/newstest2009-src.hu.sgm', 'test/newstest2009-src.en.sgm'],
        'en-hu': ['test/newstest2009-src.en.sgm', 'test/newstest2009-src.hu.sgm'],
        'it-en': ['test/newstest2009-src.it.sgm', 'test/newstest2009-src.en.sgm'],
        'en-it': ['test/newstest2009-src.en.sgm', 'test/newstest2009-src.it.sgm']
    },
    'wmt08': {
        'data': 'http://statmt.org/wmt08/test.tgz',
        'description': 'Official evaluation data.',
        'cs-en': ['test/newstest2008-src.cz.sgm', 'test/newstest2008-src.en.sgm'],
        'en-cs': ['test/newstest2008-src.en.sgm', 'test/newstest2008-src.cz.sgm'],
        'de-en': ['test/newstest2008-src.de.sgm', 'test/newstest2008-src.en.sgm'],
        'en-de': ['test/newstest2008-src.en.sgm', 'test/newstest2008-src.de.sgm'],
        'es-en': ['test/newstest2008-src.es.sgm', 'test/newstest2008-src.en.sgm'],
        'en-es': ['test/newstest2008-src.en.sgm', 'test/newstest2008-src.es.sgm'],
        'fr-en': ['test/newstest2008-src.fr.sgm', 'test/newstest2008-src.en.sgm'],
        'en-fr': ['test/newstest2008-src.en.sgm', 'test/newstest2008-src.fr.sgm'],
        'hu-en': ['test/newstest2008-src.hu.sgm', 'test/newstest2008-src.en.sgm'],
        'en-hu': ['test/newstest2008-src.en.sgm', 'test/newstest2008-src.hu.sgm']
    },
    'wmt08/nc': {
        'data': 'http://statmt.org/wmt08/test.tgz',
        'description': 'Official evaluation data (news commentary).',
        'cs-en': ['test/nc-test2008-src.cz.sgm', 'test/nc-test2008-src.en.sgm'],
        'en-cs': ['test/nc-test2008-src.en.sgm', 'test/nc-test2008-src.cz.sgm']
    },
    'wmt08/europarl': {
        'data': 'http://statmt.org/wmt08/test.tgz',
        'description': 'Official evaluation data (Europarl).',
        'de-en': ['test/test2008-src.de.sgm', 'test/test2008-src.en.sgm'],
        'en-de': ['test/test2008-src.en.sgm', 'test/test2008-src.de.sgm'],
        'es-en': ['test/test2008-src.es.sgm', 'test/test2008-src.en.sgm'],
        'en-es': ['test/test2008-src.en.sgm', 'test/test2008-src.es.sgm'],
        'fr-en': ['test/test2008-src.fr.sgm', 'test/test2008-src.en.sgm'],
        'en-fr': ['test/test2008-src.en.sgm', 'test/test2008-src.fr.sgm']
    },
}


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
    norm = norm.replace('&amp;', '"')
    norm = norm.replace('&lt;', '"')
    norm = norm.replace('&gt;', '"')

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

    def isChineseChar(uchar):
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
        else:
            return False

    sentence = sentence.strip()
    sentence_in_chars = ""
    for c in sentence:
        if isChineseChar(c):
            sentence_in_chars += " "
            sentence_in_chars += c
            sentence_in_chars += " "
        else:
            sentence_in_chars += c
    sentence = sentence_in_chars

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

    # no leading space
    sentence = re.sub(r'^\s+', r'', sentence)

    # no trailing space
    sentence = re.sub(r'\s+$', r'', sentence)

    return sentence


tokenizers = {
    '13a': tokenize_13a,
    'zh': tokenize_zh,
}


def _read(file, encoding='utf-8'):
    """Convenience function for reading compressed or plain text files.
    :param file: The file to read.
    :param encoding: The file encoding.
    """
    if file.endswith('.gz'):
        return gzip.open(file, 'rt', encoding=encoding)
    return open(file, 'rt', encoding=encoding)


def my_log(num):
    """
    Floors the log function

    :param num: the number
    :return: log(num) floored to a very low number
    """

    if num == 0.0:
        return -9999999999
    return math.log(num)


def build_signature(args, numrefs):
    """
    Builds a signature that uniquely identifies the scoring parameters used.
    :param args: the arguments passed into the script
    :return: the signature
    """

    # Abbreviations for the signature
    abbr = {
        'test': 't',
        'lang': 'l',
        'smooth': 's',
        'case': 'c',
        'tok': 'tok',
        'numrefs': '#',
        'version': 'v'
    }

    data = {'tok': args.tokenize,
            'version': VERSION}

    if args.test_set is not None:
        data['test'] = args.test_set

    if args.langpair is not None:
        data['lang'] = args.langpair

    if args.lc:
        data['case'] = 'lc'
    else:
        data['case'] = 'mixed'

    if args.smooth > 0.0:
        data['smooth'] = args.smooth

    data['numrefs'] = numrefs

    sigstr = '+'.join(['{}.{}'.format(abbr[x] if args.short else x, data[x]) for x in sorted(data.keys())])

    return sigstr


def extract_ngrams(line, max=4):
    """Extracts all the ngrams (1 <= n <= 4) from a sequence of tokens.

    :param line: a segment containing a sequence of words
    :param max: collect n-grams from 1<=n<=max
    :return: a dictionary containing ngrams and counts
    """

    ngrams = defaultdict(int)
    tokens = line.split()
    for n in range(1, max + 1):
        for i in range(0, len(tokens) - n + 1):
            ngram = ' '.join(tokens[i: i + n])
            ngrams[ngram] += 1

    return ngrams


def ref_stats(output, refs):
    ngrams = defaultdict(int)
    closest_diff = None
    closest_len = None
    for ref in refs:
        tokens = ref.split()
        reflen = len(tokens)
        diff = abs(len(output.split()) - reflen)
        if closest_diff is None or diff < closest_diff:
            closest_diff = diff
            closest_len = reflen
        elif diff == closest_diff:
            if reflen < closest_len:
                closest_len = reflen

        ngrams_ref = extract_ngrams(ref)
        for ngram in ngrams_ref.keys():
            ngrams[ngram] = max(ngrams[ngram], ngrams_ref[ngram])

    return ngrams, closest_diff, closest_len


def process_to_text(rawfile, txtfile):
    """Processes raw files to plain text files.
    :param rawfile: the input file (possibly SGML)
    :param txtfile: the plaintext file
    """

    if not os.path.exists(txtfile):
        if rawfile.endswith('.sgm') or rawfile.endswith('.sgml'):
            logging.info("Processing {} to {}".format(rawfile, txtfile))
            with _read(rawfile) as fin, open(txtfile, 'wt') as fout:
                for line in fin:
                    if line.startswith('<seg '):
                        fout.write(re.sub(r'<seg.*?>(.*)</seg>.*?', '\\1', line))


def print_test_set(test_set, langpair, side):
    """Prints to STDOUT the specified side of the specified test set
    :param test_set: the test set to print
    :param langpair: the language pair
    :param side: 'src' for source, 'ref' for reference
    """

    where = download_test_set(test_set, langpair)
    infile = where[0] if side == 'src' else where[1]
    with open(infile) as fin:
        for line in fin:
            print(line.rstrip())


def download_test_set(test_set, langpair=None):
    """Downloads the specified test to the system location specified by the SACREBLEU environment variable.
    :param test_set: the test set to download
    :param langpair: the language pair (needed for some datasets)
    :return: the set of processed files
    """

    # if not data.has_key(test_set):
    #     return None

    dataset = data[test_set]['data']
    outdir = os.path.join(SACREBLEU, test_set)
    if not os.path.exists(outdir):
        logging.info('Creating {}'.format(outdir))
        os.makedirs(outdir)

    tarball = os.path.join(outdir, os.path.basename(dataset))
    rawdir = os.path.join(outdir, 'raw')
    if not os.path.exists(tarball):
        # TODO: check MD5sum
        logging.info("Downloading {} to {}".format(dataset, tarball))
        with urllib.request.urlopen(dataset) as f, open(tarball, 'wb') as out:
            out.write(f.read())

        # Extract the tarball
        logging.info('Extracting {}'.format(tarball))
        tar = tarfile.open(tarball)
        tar.extractall(path=rawdir)

    found = []

    # Process the files into plain text
    languages = data[test_set].keys() if langpair is None else [langpair]
    for pair in languages:
        if '-' not in pair:
            continue
        src, tgt = pair.split('-')
        rawfile = os.path.join(rawdir, data[test_set][pair][0])
        outfile = os.path.join(outdir, '{}.{}'.format(pair, src))
        process_to_text(rawfile, outfile)
        found.append(outfile)

        rawfile = os.path.join(rawdir, data[test_set][pair][1])
        outfile = os.path.join(outdir, '{}.{}'.format(pair, tgt))
        process_to_text(rawfile, outfile)
        found.append(outfile)

    return found


BLEU = namedtuple('BLEU', 'score, ngram1, ngram2, ngram3, ngram4, bp, sys_len, ref_len')


def compute_bleu(instream, refstreams, smooth=0., force=False, lc=False, tokenize=False, bootstrap_trials=1) -> BLEU:
    """Produces the BLEU scores along with its sufficient statistics from a source against one or more references.

    :param instream: the input stream, one segment per line
    :param refstreams: a list of reference streams
    :param bootstrap_trials=1: number of trials for bootstrap resampling
    :return: a BLEU object containing everything you'd want
    """

    fhs = [instream] + refstreams

    # look for already-tokenized sentences
    tokenized_count = 0

    # Pre-compute segment-level data for BLEU computation.
    segmentdata = defaultdict(list)
    for sentno, lines in enumerate(zip(*fhs)):
        if lc:
            lines = [x.lower() for x in lines]

        if lines[0].rstrip().endswith(' .'):
            tokenized_count += 1

            if tokenized_count > 100 and not force:
                logging.error('FATAL: That\'s > 100 lines that end in a tokenized period (\'.\')')
                logging.error('It looks like you forgot to detokenize your test data, which will hurt your score.')
                logging.error('If you insist your data is tokenized, rerun with \'--force\'.')
                sys.exit(1)

        output, *refs = [tokenizers[tokenize](x.rstrip()) for x in lines]
        sys_ngrams = extract_ngrams(output)
        ref_ngrams, closest_diff, closest_len = ref_stats(output, refs)

        local_correct = defaultdict(int)
        local_total = defaultdict(int)

        for ngram in sys_ngrams.keys():
            n = len(ngram.split())

            local_total[n] += sys_ngrams[ngram]
            local_correct[n] += min(sys_ngrams[ngram], ref_ngrams.get(ngram, 0))

        segmentdata[sentno].append(len(output.split())) # 0: output_len
        segmentdata[sentno].append(closest_diff)        # 1: closest_diff (unused)
        segmentdata[sentno].append(closest_len)         # 2: closest_len
        segmentdata[sentno].append(local_total)         # 3: local_total
        segmentdata[sentno].append(local_correct)       # 4: local_correct

    # Based on pre-computed segment-level data, compute BLEU score for input.
    #
    # This requires seeding the RNG to get reproducible results. For now,
    # we simply freeze the seed value as 12345. This can later be changed
    # so that is is configurable. If so, the random seed needs to become
    # part of the sacreBLEU signature for future reference.
    from random import seed, randrange
    seed(12345)

    # Size of keys set equals set size
    set_size = len(segmentdata.keys())

    trial_runs = []
    for trial_run in range(bootstrap_trials):
        sys_len = 0
        ref_len = 0

        correct = defaultdict(int)
        total = defaultdict(int)

        # First trial run will always use normal test set. This results in
        # desired behaviour for bootstrap_trials=1, i.e., a single run.
        if trial_run == 0:
            input_data = segmentdata.keys()

        # Subsequent trial runs will draw with replacement from keys set.
        else:
            input_data = (randrange(0, set_size-1) for _ in range(set_size))

        # Compute BLEU score for current trial, based on pre-computed data.
        for sentno in input_data:
            output_len = segmentdata[sentno][0]
            closest_diff = segmentdata[sentno][1]
            closest_len = segmentdata[sentno][2]
            local_total = segmentdata[sentno][3]
            local_correct = segmentdata[sentno][4]

            sys_len += output_len
            ref_len += closest_len

            for n in local_total.keys():
                total[n] += local_total[n]
                correct[n] += local_correct[n]

        if sum(total) == 0:
            logging.error('No input?')
            sys.exit(1)

        precisions = [0, 0, 0, 0, 0]

        for n in range(1, 5):
            precisions[n] = max(smooth, 100. * correct[n] / total[n] if total.get(n) > 0 else 0)

        brevity_penalty = 1.0
        if sys_len < ref_len:
            brevity_penalty = math.exp(1 - ref_len / sys_len)

        bleu = 1. * brevity_penalty * math.exp(sum(map(my_log, precisions[1:])) / 4)
        trial_runs.append([bleu, precisions[1], precisions[2], precisions[3], precisions[4], brevity_penalty, sys_len, ref_len])

    # Compute average BLEU score and component values.
    avgBleu = [
      sum(x[0] for x in trial_runs) / len(trial_runs),      # bleu
      sum(x[1] for x in trial_runs) / len(trial_runs),      # precisions[1]
      sum(x[2] for x in trial_runs) / len(trial_runs),      # precisions[2]
      sum(x[3] for x in trial_runs) / len(trial_runs),      # precisions[3]
      sum(x[4] for x in trial_runs) / len(trial_runs),      # precisions[4]
      sum(x[5] for x in trial_runs) / len(trial_runs),      # brevity_penalty
      int(sum(x[6] for x in trial_runs) / len(trial_runs)), # sys_len
      int(sum(x[7] for x in trial_runs) / len(trial_runs)), # ref_len
    ]

    if bootstrap_trials > 1:
        print('Bootstrap trials: n={0}'.format(bootstrap_trials))
        allBleuScores = [x[0] for x in trial_runs]
        try:
            from numpy import mean, std
            from math import sqrt

            # Compute 0.95 confidence interval around BLEU score mean.
            xbar = mean(allBleuScores)
            s = std(allBleuScores)
            sqrtn = sqrt(bootstrap_trials)
            z = 1.96
            confidenceInterval = z * s / sqrtn

        except ImportError:
            logger.error('Could not import numpy for confidence interval computation')
            xbar = sum(allBleuScores) / len(allBleuScores)
            confidenceInterval = None

        finally:
            if confidenceInterval:
                print('Mean BLEU score: {0:.2f} +/- {1:.2f}'.format(xbar, confidenceInterval))
            else:
                print('Mean BLEU score: {0:.2f}'.format(xbar))

    return BLEU._make(avgBleu)


def main():
    arg_parser = argparse.ArgumentParser(description='sacréBLEU: Hassle-free computation of shareable BLEU scores.'
                                         'Quick usage: score your detokenized output against WMT\'14 EN-DE:'
                                         '    cat output.detok.de | ./sacreBLEU -t wmt14 -l en-de')
    arg_parser.add_argument('--test-set', '-t', type=str, default=None,
                            choices=data.keys(),
                            help='The test set to use')
    arg_parser.add_argument('-lc', action='store_true', default=False,
                            help='Use case-insensitive BLEU (default: actual case)')
    arg_parser.add_argument('--smooth', '-s', type=float, default=0.0,
                            help='Smooth zero-count precisions with specified value')
    arg_parser.add_argument('--tokenize', '-tok', choices=['13a', 'zh'], default='13a',
                            help='Tokenization method to use.')
    arg_parser.add_argument('--language-pair', '-l', dest='langpair', default=None,
                            help='source-target language pair (2-char ISO639-1 codes)')
    arg_parser.add_argument('--download', type=str, default=None,
                            help='Download a test set and quit')
    arg_parser.add_argument('--echo', choices=['src', 'ref'], type=str, default=None,
                            help='Output the source or reference to STDOUT and quit.')
    arg_parser.add_argument('refs', nargs='*', default=[],
                            help='Optional list of references (for backwards-compatibility with older scripts).')
    arg_parser.add_argument('--short', default=False, action='store_true',
                            help='Produce a shorter (less human readable) signature.')
    arg_parser.add_argument('--force', default=False, action='store_true',
                            help='Insist that your tokenized input is actually detokenized.')
    arg_parser.add_argument('--quiet', '-q', default=False, action='store_true',
                            help='Suppress informative output.')
    arg_parser.add_argument('--encoding', '-e', type=str, default='utf-8',
                            help='Open text files with specified encoding (default: %(default)s)')
    arg_parser.add_argument('--bootstrap-trials', '-b', type=int, default=1,
                            help='Compute BLEU based on bootstrap resampling with n trials (default: %(default)d)')
    arg_parser.add_argument('-V', '--version', action='version',
                            version='%(prog)s {}'.format(VERSION))
    args = arg_parser.parse_args()

    if not args.quiet:
        logging.basicConfig(level=logging.INFO, format='sacréBLEU: %(message)s')

    if args.download:
        download_test_set(args.download, args.langpair)
        sys.exit(0)

    if args.test_set is not None and args.test_set not in data:
        logging.error('The available test sets are: ')
        for ts in sorted(data.keys(), reverse=True):
            logging.error('  {}: {}'.format(ts, data[ts].get('description', '')))
        sys.exit(1)

    if args.test_set and (args.langpair is None or args.langpair not in data[args.test_set]):
        logging.error('I need a language pair (-l).')
        logging.error('Available language pairs for test set "{}": {}'.format(args.test_set, ', '.join(filter(lambda x: '-' in x, data[args.test_set].keys()))))
        sys.exit(1)

    if args.echo:
        if args.langpair is None or args.test_set is None:
            logging.warn("--echo requires a test set (--t) and a language pair (-l)")
            sys.exit(1)
        print_test_set(args.test_set, args.langpair, args.echo)
        sys.exit(0)

    if args.test_set is None and len(args.refs) == 0:
        logging.error('I need either -t (test set) or a list of references')
        logging.error('The available test sets are: ')
        for ts in sorted(data.keys(), reverse=True):
            logging.error('  {}: {}'.format(ts, data[ts].get('description', '')))
        sys.exit(1)
    elif args.test_set is not None and len(args.refs) > 0:
        logging.error('I need x-either a test set (-t) or a list of references')
        sys.exit(1)

    if args.test_set:
        src, ref = download_test_set(args.test_set, args.langpair)
        refs = [ref]
    else:
        refs = args.refs

    # Read references
    refs = [_read(x, args.encoding) for x in refs]

    if args.langpair is not None:
        source, target = args.langpair.split('-')
        if target == 'zh' and args.tokenize != 'zh':
            logging.warn('You should also pass "--tok zh" when scoring Chinese...')

    bleu = compute_bleu(sys.stdin, refs, smooth=args.smooth, force=args.force,
                        lc=args.lc, tokenize=args.tokenize, bootstrap_trials=args.bootstrap_trials)

    version_str = build_signature(args, len(refs))

    print('BLEU+{} = {:.2f} {:.1f}/{:.1f}/{:.1f}/{:.1f} (BP = {:.3f} ratio = {:.3f} hyp_len = {:d} ref_len = {:d})'.format(version_str, bleu.score, bleu.ngram1, bleu.ngram2, bleu.ngram3, bleu.ngram4, bleu.bp, bleu.sys_len / bleu.ref_len, bleu.sys_len, bleu.ref_len))


if __name__ == '__main__':
    main()
