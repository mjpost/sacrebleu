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

"""
SacreBLEU provides hassle-free computation of shareable, comparable, and reproducible BLEU scores.
Inspired by Rico Sennrich's `multi-bleu-detok.perl`, it produces the official WMT scores but works with plain text.
It also knows all the standard test sets and handles downloading, processing, and tokenization for you.

See the [README.md] file for more information.
"""

import argparse
import gzip
import hashlib
import io
import logging
import math
import os
import portalocker
import re
import sys
import ssl
import urllib.request

from collections import Counter
from itertools import zip_longest
from typing import List, Iterable, Tuple, Union
from .tokenizer import TOKENIZERS, TokenizeMeCab
from .dataset import DATASETS, DOMAINS, COUNTRIES, SUBSETS
from . import __version__ as VERSION

try:
    # SIGPIPE is not available on Windows machines, throwing an exception.
    from signal import SIGPIPE

    # If SIGPIPE is available, change behaviour to default instead of ignore.
    from signal import signal, SIG_DFL
    signal(SIGPIPE, SIG_DFL)

except ImportError:
    logging.warning('Could not import signal.SIGPIPE (this is expected on Windows machines)')

# Where to store downloaded test sets.
# Define the environment variable $SACREBLEU, or use the default of ~/.sacrebleu.
#
# Querying for a HOME environment variable can result in None (e.g., on Windows)
# in which case the os.path.join() throws a TypeError. Using expanduser() is
# a safe way to get the user's home folder.
USERHOME = os.path.expanduser("~")
SACREBLEU_DIR = os.environ.get('SACREBLEU', os.path.join(USERHOME, '.sacrebleu'))

# n-gram order. Don't change this.
NGRAM_ORDER = 4

# Default values for CHRF
CHRF_ORDER = 6
# default to 2 (per http://www.aclweb.org/anthology/W16-2341)
CHRF_BETA = 2

# The default floor value to use with `--smooth floor`
SMOOTH_VALUE_DEFAULT = {'floor': 0.0, 'add-k': 1}


DEFAULT_TOKENIZER = '13a'


def smart_open(file, mode='rt', encoding='utf-8'):
    """Convenience function for reading compressed or plain text files.
    :param file: The file to read.
    :param mode: The file mode (read, write).
    :param encoding: The file encoding.
    """
    if file.endswith('.gz'):
        return gzip.open(file, mode=mode, encoding=encoding, newline="\n")
    return open(file, mode=mode, encoding=encoding, newline="\n")


def my_log(num):
    """
    Floors the log function

    :param num: the number
    :return: log(num) floored to a very low number
    """

    if num == 0.0:
        return -9999999999
    return math.log(num)


def bleu_signature(args, numrefs):
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
        'version': 'v',
        'origlang': 'o',
        'subset': 'S',
    }

    signature = {'tok': args.tokenize,
                 'version': VERSION,
                 'smooth': args.smooth,
                 'numrefs': numrefs,
                 'case': 'lc' if args.lc else 'mixed'}

    # For the Japanese tokenizer, add a dictionary type and its version to the signature.
    if args.tokenize == "ja-mecab":
        signature['tok'] += "-" + TokenizeMeCab().signature()

    if args.test_set is not None:
        signature['test'] = args.test_set

    if args.langpair is not None:
        signature['lang'] = args.langpair

    if args.origlang is not None:
        signature['origlang'] = args.origlang
    if args.subset is not None:
        signature['subset'] = args.subset

    sigstr = '+'.join(['{}.{}'.format(abbr[x] if args.short else x, signature[x]) for x in sorted(signature.keys())])

    return sigstr


def chrf_signature(args, numrefs):
    """
    Builds a signature that uniquely identifies the scoring parameters used.
    :param args: the arguments passed into the script
    :return: the chrF signature
    """

    # Abbreviations for the signature
    abbr = {
        'test': 't',
        'lang': 'l',
        'numchars': 'n',
        'space': 's',
        'case': 'c',
        'numrefs': '#',
        'version': 'v',
        'origlang': 'o',
        'subset': 'S',
    }

    signature = {'version': VERSION,
                 'space': args.chrf_whitespace,
                 'numchars': args.chrf_order,
                 'numrefs': numrefs,
                 'case': 'lc' if args.lc else 'mixed'}

    if args.test_set is not None:
        signature['test'] = args.test_set

    if args.langpair is not None:
        signature['lang'] = args.langpair

    if args.origlang is not None:
        signature['origlang'] = args.origlang
    if args.subset is not None:
        signature['subset'] = args.subset

    sigstr = '+'.join(['{}.{}'.format(abbr[x] if args.short else x, signature[x]) for x in sorted(signature.keys())])

    return sigstr


def extract_ngrams(line, min_order=1, max_order=NGRAM_ORDER) -> Counter:
    """Extracts all the ngrams (min_order <= n <= max_order) from a sequence of tokens.

    :param line: A segment containing a sequence of words.
    :param min_order: Minimum n-gram length (default: 1).
    :param max_order: Maximum n-gram length (default: NGRAM_ORDER).
    :return: a dictionary containing ngrams and counts
    """

    ngrams = Counter()
    tokens = line.split()
    for n in range(min_order, max_order + 1):
        for i in range(0, len(tokens) - n + 1):
            ngram = ' '.join(tokens[i: i + n])
            ngrams[ngram] += 1

    return ngrams


def extract_char_ngrams(s: str, n: int) -> Counter:
    """
    Yields counts of character n-grams from string s of order n.
    """
    return Counter([s[i:i + n] for i in range(len(s) - n + 1)])


def ref_stats(output, refs):
    ngrams = Counter()
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


def _clean(s):
    """
    Removes trailing and leading spaces and collapses multiple consecutive internal spaces to a single one.

    :param s: The string.
    :return: A cleaned-up string.
    """
    return re.sub(r'\s+', ' ', s.strip())


def process_to_text(rawfile, txtfile, field: int=None):
    """Processes raw files to plain text files.
    :param rawfile: the input file (possibly SGML)
    :param txtfile: the plaintext file
    :param field: For TSV files, which field to extract.
    """

    if not os.path.exists(txtfile) or os.path.getsize(txtfile) == 0:
        logging.info("Processing %s to %s", rawfile, txtfile)
        if rawfile.endswith('.sgm') or rawfile.endswith('.sgml'):
            with smart_open(rawfile) as fin, smart_open(txtfile, 'wt') as fout:
                for line in fin:
                    if line.startswith('<seg '):
                        print(_clean(re.sub(r'<seg.*?>(.*)</seg>.*?', '\\1', line)), file=fout)
        elif rawfile.endswith('.xml'): # IWSLT
            with smart_open(rawfile) as fin, smart_open(txtfile, 'wt') as fout:
                for line in fin:
                    if line.startswith('<seg '):
                        print(_clean(re.sub(r'<seg.*?>(.*)</seg>.*?', '\\1', line)), file=fout)
        elif rawfile.endswith('.txt'): # wmt17/ms
            with smart_open(rawfile) as fin, smart_open(txtfile, 'wt') as fout:
                for line in fin:
                    print(line.rstrip(), file=fout)
        elif rawfile.endswith('.tsv'): # MTNT
            with smart_open(rawfile) as fin, smart_open(txtfile, 'wt') as fout:
                for line in fin:
                    print(line.rstrip().split('\t')[field], file=fout)


def print_test_set(test_set, langpair, side, origlang=None, subset=None):
    """Prints to STDOUT the specified side of the specified test set
    :param test_set: the test set to print
    :param langpair: the language pair
    :param side: 'src' for source, 'ref' for reference
    :param origlang: print only sentences with a given original language (2-char ISO639-1 code), "non-" prefix means negation
    :param subset: print only sentences whose document annotation matches a given regex
    """

    files = download_test_set(test_set, langpair)
    if side == 'src':
        files = [files[0]]
    elif side == 'ref':
        files.pop(0)

    streams = [smart_open(file) for file in files]
    streams = _filter_subset(streams, test_set, langpair, origlang, subset)
    for lines in zip(*streams):
        print('\t'.join(map(lambda x: x.rstrip(), lines)))


def download_test_set(test_set, langpair=None):
    """Downloads the specified test to the system location specified by the SACREBLEU environment variable.

    :param test_set: the test set to download
    :param langpair: the language pair (needed for some datasets)
    :return: the set of processed files
    """

    outdir = os.path.join(SACREBLEU_DIR, test_set)
    os.makedirs(outdir, exist_ok=True)

    expected_checksums = DATASETS[test_set].get('md5', [None] * len(DATASETS[test_set]))
    for dataset, expected_md5 in zip(DATASETS[test_set]['data'], expected_checksums):
        tarball = os.path.join(outdir, os.path.basename(dataset))
        rawdir = os.path.join(outdir, 'raw')

        lockfile = '{}.lock'.format(tarball)
        with portalocker.Lock(lockfile, 'w', timeout=60):
            if not os.path.exists(tarball) or os.path.getsize(tarball) == 0:
                logging.info("Downloading %s to %s", dataset, tarball)
                try:
                    with urllib.request.urlopen(dataset) as f, open(tarball, 'wb') as out:
                        out.write(f.read())
                except ssl.SSLError:
                    logging.warning('An SSL error was encountered in downloading the files. If you\'re on a Mac, '
                                'you may need to run the "Install Certificates.command" file located in the '
                                '"Python 3" folder, often found under /Applications')
                    sys.exit(1)

                # Check md5sum
                if expected_md5 is not None:
                    md5 = hashlib.md5()
                    with open(tarball, 'rb') as infile:
                        for line in infile:
                            md5.update(line)
                    if md5.hexdigest() != expected_md5:
                        logging.error('Fatal: MD5 sum of downloaded file was incorrect (got {}, expected {}).'.format(md5.hexdigest(), expected_md5))
                        logging.error('Please manually delete "{}" and rerun the command.'.format(tarball))
                        logging.error('If the problem persists, the tarball may have changed, in which case, please contact the SacreBLEU maintainer.')
                        sys.exit(1)
                    else:
                        logging.info('Checksum passed: {}'.format(md5.hexdigest()))

                # Extract the tarball
                logging.info('Extracting %s', tarball)
                if tarball.endswith('.tar.gz') or tarball.endswith('.tgz'):
                    import tarfile
                    with tarfile.open(tarball) as tar:
                        tar.extractall(path=rawdir)
                elif tarball.endswith('.zip'):
                    import zipfile
                    with zipfile.ZipFile(tarball, 'r') as zipfile:
                        zipfile.extractall(path=rawdir)

    found = []

    # Process the files into plain text
    languages = DATASETS[test_set].keys() if langpair is None else [langpair]
    for pair in languages:
        if '-' not in pair:
            continue
        src, tgt = pair.split('-')
        rawfile = DATASETS[test_set][pair][0]
        field = None  # used for TSV files
        if rawfile.endswith('.tsv'):
            field, rawfile = rawfile.split(':', maxsplit=1)
            field = int(field)
        rawpath = os.path.join(rawdir, rawfile)
        outpath = os.path.join(outdir, '{}.{}'.format(pair, src))
        process_to_text(rawpath, outpath, field=field)
        found.append(outpath)

        refs = DATASETS[test_set][pair][1:]
        for i, ref in enumerate(refs):
            field = None
            if ref.endswith('.tsv'):
                field, ref = ref.split(':', maxsplit=1)
                field = int(field)
            rawpath = os.path.join(rawdir, ref)
            if len(refs) >= 2:
                outpath = os.path.join(outdir, '{}.{}.{}'.format(pair, tgt, i))
            else:
                outpath = os.path.join(outdir, '{}.{}'.format(pair, tgt))
            process_to_text(rawpath, outpath, field=field)
            found.append(outpath)

    return found


class Result:
    def __init__(self, score: float):
        self.score = score

    def __str__(self):
        return self.format()


class BLEU(Result):
    def __init__(self,
                 score: float,
                 counts,
                 totals,
                 precisions,
                 bp,
                 sys_len,
                 ref_len):
        super().__init__(score)

        self.counts = counts
        self.totals = totals
        self.precisions = precisions
        self.bp = bp
        self.sys_len = sys_len
        self.ref_len = ref_len

    def format(self, width=2):
        precisions = "/".join(["{:.1f}".format(p) for p in self.precisions])
        return 'BLEU = {score:.{width}f} {precisions} (BP = {bp:.3f} ratio = {ratio:.3f} hyp_len = {sys_len:d} ref_len = {ref_len:d})'.format(
            score=self.score,
            width=width,
            precisions=precisions,
            bp=self.bp,
            ratio=self.sys_len / self.ref_len,
            sys_len=self.sys_len,
            ref_len=self.ref_len)


class CHRF(Result):
    def __init__(self, score: float):
        super().__init__(score)

    def format(self, width=2):
        return '{score:.{width}f}'.format(score=self.score, width=width)


def compute_bleu(correct: List[int],
                 total: List[int],
                 sys_len: int,
                 ref_len: int,
                 smooth_method = 'none',
                 smooth_value = None,
                 use_effective_order = False) -> BLEU:
    """Computes BLEU score from its sufficient statistics. Adds smoothing.

    Smoothing methods (citing "A Systematic Comparison of Smoothing Techniques for Sentence-Level BLEU",
    Boxing Chen and Colin Cherry, WMT 2014: http://aclweb.org/anthology/W14-3346)

    - exp: NIST smoothing method (Method 3)
    - floor: Method 1
    - add-k: Method 2 (generalizing Lin and Och, 2004)
    - none: do nothing.

    :param correct: List of counts of correct ngrams, 1 <= n <= NGRAM_ORDER
    :param total: List of counts of total ngrams, 1 <= n <= NGRAM_ORDER
    :param sys_len: The cumulative system length
    :param ref_len: The cumulative reference length
    :param smooth: The smoothing method to use
    :param smooth_value: The smoothing value added, if smooth method 'floor' is used
    :param use_effective_order: If true, use the length of `correct` for the n-gram order instead of NGRAM_ORDER.
    :return: A BLEU object with the score (100-based) and other statistics.
    """
    if smooth_method in SMOOTH_VALUE_DEFAULT and smooth_value is None:
        smooth_value = SMOOTH_VALUE_DEFAULT[smooth_method]

    precisions = [0 for x in range(NGRAM_ORDER)]

    smooth_mteval = 1.
    effective_order = NGRAM_ORDER
    for n in range(1, NGRAM_ORDER + 1):
        if smooth_method == 'add-k' and n > 1:
            correct[n-1] += smooth_value
            total[n-1] += smooth_value
        if total[n-1] == 0:
            break

        if use_effective_order:
            effective_order = n

        if correct[n-1] == 0:
            if smooth_method == 'exp':
                smooth_mteval *= 2
                precisions[n-1] = 100. / (smooth_mteval * total[n-1])
            elif smooth_method == 'floor':
                precisions[n-1] = 100. * smooth_value / total[n-1]
        else:
            precisions[n-1] = 100. * correct[n-1] / total[n-1]

    # If the system guesses no i-grams, 1 <= i <= NGRAM_ORDER, the BLEU score is 0 (technically undefined).
    # This is a problem for sentence-level BLEU or a corpus of short sentences, where systems will get no credit
    # if sentence lengths fall under the NGRAM_ORDER threshold. This fix scales NGRAM_ORDER to the observed
    # maximum order. It is only available through the API and off by default

    brevity_penalty = 1.0
    if sys_len < ref_len:
        brevity_penalty = math.exp(1 - ref_len / sys_len) if sys_len > 0 else 0.0

    score = brevity_penalty * math.exp(sum(map(my_log, precisions[:effective_order])) / effective_order)

    return BLEU(score, correct, total, precisions, brevity_penalty, sys_len, ref_len)


def sentence_bleu(hypothesis: str,
                  references: List[str],
                  smooth_method: str = 'floor',
                  smooth_value: float = None,
                  use_effective_order: bool = True) -> BLEU:
    """
    Computes BLEU on a single sentence pair.

    Disclaimer: computing BLEU on the sentence level is not its intended use,
    BLEU is a corpus-level metric.

    :param hypothesis: Hypothesis string.
    :param reference: Reference string.
    :param smooth_value: For 'floor' smoothing, the floor value to use.
    :param use_effective_order: Account for references that are shorter than the largest n-gram.
    :return: Returns a single BLEU score as a float.
    """
    bleu = corpus_bleu(hypothesis, references,
                       smooth_method=smooth_method,
                       smooth_value=smooth_value,
                       use_effective_order=use_effective_order)
    return bleu


def corpus_bleu(sys_stream: Union[str, Iterable[str]],
                ref_streams: Union[str, List[Iterable[str]]],
                smooth_method='exp',
                smooth_value=None,
                force=False,
                lowercase=False,
                tokenize=DEFAULT_TOKENIZER,
                use_effective_order=False) -> BLEU:
    """Produces BLEU scores along with its sufficient statistics from a source against one or more references.

    :param sys_stream: The system stream (a sequence of segments)
    :param ref_streams: A list of one or more reference streams (each a sequence of segments)
    :param smooth: The smoothing method to use
    :param smooth_value: For 'floor' smoothing, the floor to use
    :param force: Ignore data that looks already tokenized
    :param lowercase: Lowercase the data
    :param tokenize: The tokenizer to use
    :return: a BLEU object containing everything you'd want
    """

    # Add some robustness to the input arguments
    if isinstance(sys_stream, str):
        sys_stream = [sys_stream]
    if isinstance(ref_streams, str):
        ref_streams = [[ref_streams]]

    sys_len = 0
    ref_len = 0

    correct = [0 for n in range(NGRAM_ORDER)]
    total = [0 for n in range(NGRAM_ORDER)]

    # look for already-tokenized sentences
    tokenized_count = 0

    fhs = [sys_stream] + ref_streams
    for lines in zip_longest(*fhs):
        if None in lines:
            raise EOFError("Source and reference streams have different lengths!")

        if lowercase:
            lines = [x.lower() for x in lines]

        if not (force or tokenize == 'none') and lines[0].rstrip().endswith(' .'):
            tokenized_count += 1

            if tokenized_count == 100:
                logging.warning('That\'s 100 lines that end in a tokenized period (\'.\')')
                logging.warning('It looks like you forgot to detokenize your test data, which may hurt your score.')
                logging.warning('If you insist your data is detokenized, or don\'t care, you can suppress this message with \'--force\'.')

        output, *refs = [TOKENIZERS[tokenize](x.rstrip()) for x in lines]

        ref_ngrams, closest_diff, closest_len = ref_stats(output, refs)

        sys_len += len(output.split())
        ref_len += closest_len

        sys_ngrams = extract_ngrams(output)
        for ngram in sys_ngrams.keys():
            n = len(ngram.split())
            correct[n-1] += min(sys_ngrams[ngram], ref_ngrams.get(ngram, 0))
            total[n-1] += sys_ngrams[ngram]

    return compute_bleu(correct, total, sys_len, ref_len, smooth_method=smooth_method, smooth_value=smooth_value, use_effective_order=use_effective_order)


def raw_corpus_bleu(sys_stream,
                    ref_streams,
                    smooth_value=None) -> BLEU:
    """Convenience function that wraps corpus_bleu().
    This is convenient if you're using sacrebleu as a library, say for scoring on dev.
    It uses no tokenization and 'floor' smoothing, with the floor default to 0 (no smoothing).

    :param sys_stream: the system stream (a sequence of segments)
    :param ref_streams: a list of one or more reference streams (each a sequence of segments)
    """
    return corpus_bleu(sys_stream, ref_streams, smooth_method='floor', smooth_value=smooth_value, force=True, tokenize='none', use_effective_order=True)


def delete_whitespace(text: str) -> str:
    """
    Removes whitespaces from text.
    """
    return re.sub(r'\s+', '', text).strip()


def get_sentence_statistics(hypothesis: str,
                            reference: str,
                            order: int = CHRF_ORDER,
                            remove_whitespace: bool = True) -> List[float]:
    hypothesis = delete_whitespace(hypothesis) if remove_whitespace else hypothesis
    reference = delete_whitespace(reference) if remove_whitespace else reference
    statistics = [0] * (order * 3)
    for i in range(order):
        n = i + 1
        hypothesis_ngrams = extract_char_ngrams(hypothesis, n)
        reference_ngrams = extract_char_ngrams(reference, n)
        common_ngrams = hypothesis_ngrams & reference_ngrams
        statistics[3 * i + 0] = sum(hypothesis_ngrams.values())
        statistics[3 * i + 1] = sum(reference_ngrams.values())
        statistics[3 * i + 2] = sum(common_ngrams.values())
    return statistics


def get_corpus_statistics(hypotheses: Iterable[str],
                          references: Iterable[str],
                          order: int = CHRF_ORDER,
                          remove_whitespace: bool = True) -> List[float]:
    corpus_statistics = [0] * (order * 3)
    for hypothesis, reference in zip(hypotheses, references):
        statistics = get_sentence_statistics(hypothesis, reference, order=order, remove_whitespace=remove_whitespace)
        for i in range(len(statistics)):
            corpus_statistics[i] += statistics[i]
    return corpus_statistics


def _avg_precision_and_recall(statistics: List[float], order: int) -> Tuple[float, float]:
    avg_precision = 0.0
    avg_recall = 0.0
    effective_order = 0
    for i in range(order):
        hypotheses_ngrams = statistics[3 * i + 0]
        references_ngrams = statistics[3 * i + 1]
        common_ngrams = statistics[3 * i + 2]
        if hypotheses_ngrams > 0 and references_ngrams > 0:
            avg_precision += common_ngrams / hypotheses_ngrams
            avg_recall += common_ngrams / references_ngrams
            effective_order += 1
    if effective_order == 0:
        return 0.0, 0.0
    avg_precision /= effective_order
    avg_recall /= effective_order
    return avg_precision, avg_recall


def _chrf(avg_precision, avg_recall, beta: int = CHRF_BETA) -> float:
    if avg_precision + avg_recall == 0:
        return 0.0
    beta_square = beta ** 2
    score = (1 + beta_square) * (avg_precision * avg_recall) / ((beta_square * avg_precision) + avg_recall)
    return score


def corpus_chrf(hypotheses: Iterable[str],
                references: Iterable[str],
                order: int = CHRF_ORDER,
                beta: float = CHRF_BETA,
                remove_whitespace: bool = True) -> CHRF:
    """
    Computes Chrf on a corpus.

    :param hypotheses: Stream of hypotheses.
    :param references: Stream of references
    :param order: Maximum n-gram order.
    :param remove_whitespace: Whether to delete all whitespace from hypothesis and reference strings.
    :param beta: Defines importance of recall w.r.t precision. If beta=1, same importance.
    :return: Chrf score.
    """
    corpus_statistics = get_corpus_statistics(hypotheses, references, order=order, remove_whitespace=remove_whitespace)
    avg_precision, avg_recall = _avg_precision_and_recall(corpus_statistics, order)
    return CHRF(_chrf(avg_precision, avg_recall, beta=beta))


def sentence_chrf(hypothesis: str,
                  reference: str,
                  order: int = CHRF_ORDER,
                  beta: float = CHRF_BETA,
                  remove_whitespace: bool = True) -> CHRF:
    """
    Computes ChrF on a single sentence pair.

    :param hypothesis: Hypothesis string.
    :param reference: Reference string.
    :param order: Maximum n-gram order.
    :param remove_whitespace: Whether to delete whitespaces from hypothesis and reference strings.
    :param beta: Defines importance of recall w.r.t precision. If beta=1, same importance.
    :return: Chrf score.
    """
    statistics = get_sentence_statistics(hypothesis, reference, order=order, remove_whitespace=remove_whitespace)
    avg_precision, avg_recall = _avg_precision_and_recall(statistics, order)
    return CHRF(_chrf(avg_precision, avg_recall, beta=beta))


def get_langpairs_for_testset(testset: str) -> List:
    """Return a list of language pairs for a given test set."""
    return list(filter(lambda x: re.match('\w\w\-\w\w', x), DATASETS.get(testset, {}).keys()))


def get_a_list_of_testset_names() -> str:
    """Return a string with a formatted list of available test sets plus their descriptions. """
    message = 'The available test sets are:'
    for testset in sorted(DATASETS.keys(), reverse=True):
        message += '\n%20s: %s' % (testset, DATASETS[testset].get('description', ''))
    return message


def _available_origlangs(test_sets, langpair):
    """Return a list of origlang values in according to the raw SGM files."""
    origlangs = set()
    for test_set in test_sets.split(','):
        rawfile = os.path.join(SACREBLEU_DIR, test_set, 'raw', DATASETS[test_set][langpair][0])
        if rawfile.endswith('.sgm'):
            with smart_open(rawfile) as fin:
                for line in fin:
                    if line.startswith('<doc '):
                        doc_origlang = re.sub(r'.* origlang="([^"]+)".*\n', '\\1', line)
                        origlangs.add(doc_origlang)
    return sorted(list(origlangs))


def _filter_subset(systems, test_sets, langpair, origlang, subset=None):
    """Filter sentences with a given origlang (or subset) according to the raw SGM files."""
    if origlang is None and subset is None:
        return systems
    if test_sets is None or langpair is None:
        raise ValueError('Filtering for --origlang or --subset needs a test (-t) and a language pair (-l).')

    indices_to_keep = []
    for test_set in test_sets.split(','):
        rawfile = os.path.join(SACREBLEU_DIR, test_set, 'raw', DATASETS[test_set][langpair][0])
        if not rawfile.endswith('.sgm'):
            raise Exception('--origlang and --subset supports only *.sgm files, not %s', rawfile)
        if subset is not None:
            if test_set not in SUBSETS:
                raise Exception('No subset annotation available for test set ' + test_set)
            doc_to_tags = SUBSETS[test_set]
        number_sentences_included = 0
        with smart_open(rawfile) as fin:
            include_doc = False
            for line in fin:
                if line.startswith('<doc '):
                    if origlang is None:
                        include_doc = True
                    else:
                        doc_origlang = re.sub(r'.* origlang="([^"]+)".*\n', '\\1', line)
                        if origlang.startswith('non-'):
                            include_doc = doc_origlang != origlang[4:]
                        else:
                            include_doc = doc_origlang == origlang
                    if subset is not None:
                        doc_id = re.sub(r'.* docid="([^"]+)".*\n', '\\1', line)
                        if not re.search(subset, doc_to_tags.get(doc_id, '')):
                            include_doc = False
                if line.startswith('<seg '):
                    indices_to_keep.append(include_doc)
                    number_sentences_included += 1 if include_doc else 0
    return [[sentence for sentence,keep in zip(sys, indices_to_keep) if keep] for sys in systems]


def main():
    args = parse_args()

    # Explicitly set the encoding
    sys.stdin = open(sys.stdin.fileno(), mode='r', encoding='utf-8', buffering=True, newline="\n")
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=True)

    if not args.quiet:
        logging.basicConfig(level=logging.INFO, format='sacreBLEU: %(message)s')

    if args.download:
        download_test_set(args.download, args.langpair)
        sys.exit(0)

    if args.list:
        if args.test_set:
            print(' '.join(get_langpairs_for_testset(args.test_set)))
        else:
            print(get_a_list_of_testset_names())
        sys.exit(0)

    if args.sentence_level and len(args.metrics) > 1:
        logging.error('Only one metric can be used with Sentence-level reporting.')
        sys.exit(1)

    if args.citation:
        if not args.test_set:
            logging.error('I need a test set (-t).')
            sys.exit(1)
        for test_set in args.test_set.split(','):
            if 'citation' not in DATASETS[test_set]:
                logging.error('No citation found for %s', test_set)
            else:
                print(DATASETS[test_set]['citation'])
        sys.exit(0)

    if args.num_refs != 1 and (args.test_set is not None or len(args.refs) > 1):
        logging.error('The --num-refs argument allows you to provide any number of tab-delimited references in a single file.')
        logging.error('You can only use it with externaly-provided references, however (i.e., not with `-t`),')
        logging.error('and you cannot then provide multiple reference files.')
        sys.exit(1)

    if args.test_set is not None:
        for test_set in args.test_set.split(','):
            if test_set not in DATASETS:
                logging.error('Unknown test set "%s"\n%s', test_set, get_a_list_of_testset_names())
                sys.exit(1)

    if args.test_set is None:
        if len(args.refs) == 0:
            logging.error('I need either a predefined test set (-t) or a list of references')
            logging.error(get_a_list_of_testset_names())
            sys.exit(1)
    elif len(args.refs) > 0:
        logging.error('I need exactly one of (a) a predefined test set (-t) or (b) a list of references')
        sys.exit(1)
    elif args.langpair is None:
        logging.error('I need a language pair (-l).')
        sys.exit(1)
    else:
        for test_set in args.test_set.split(','):
            if args.langpair not in DATASETS[test_set]:
                logging.error('No such language pair "%s"', args.langpair)
                logging.error('Available language pairs for test set "%s": %s', test_set,
                      ', '.join(x for x in DATASETS[test_set].keys() if '-' in x))
                sys.exit(1)

    if args.echo:
        if args.langpair is None or args.test_set is None:
            logging.warning("--echo requires a test set (--t) and a language pair (-l)")
            sys.exit(1)
        for test_set in args.test_set.split(','):
            print_test_set(test_set, args.langpair, args.echo, args.origlang, args.subset)
        sys.exit(0)

    if args.test_set is not None and args.tokenize == 'none':
        logging.warning("You are turning off sacrebleu's internal tokenization ('--tokenize none'), presumably to supply\n"
                        "your own reference tokenization. Published numbers will not be comparable with other papers.\n")

    # Internal tokenizer settings. Set to 'zh' for Chinese  DEFAULT_TOKENIZER (
    if args.tokenize is None:
        # set default
        if args.langpair is not None and args.langpair.split('-')[1] == 'zh':
            args.tokenize = 'zh'
        elif args.langpair is not None and args.langpair.split('-')[1] == 'ja':
            args.tokenize = 'ja-mecab'
        else:
            args.tokenize = DEFAULT_TOKENIZER

    if args.langpair is not None and 'bleu' in args.metrics:
        if args.langpair.split('-')[1] == 'zh' and args.tokenize != 'zh':
            logging.warning('You should also pass "--tok zh" when scoring Chinese...')
        if args.langpair.split('-')[1] == 'ja' and not args.tokenize.startswith('ja-'):
            logging.warning('You should also pass "--tok ja-mecab" when scoring Japanese...')

    # concat_ref_files is a list of list of reference filenames, for example:
    # concat_ref_files = [[testset1_refA, testset1_refB], [testset2_refA, testset2_refB]]
    if args.test_set is None:
        concat_ref_files = [args.refs]
    else:
        concat_ref_files = []
        for test_set in args.test_set.split(','):
            _, *ref_files = download_test_set(test_set, args.langpair)
            if len(ref_files) == 0:
                logging.warning('No references found for test set {}/{}.'.format(test_set, args.langpair))
            concat_ref_files.append(ref_files)


    inputfh = io.TextIOWrapper(sys.stdin.buffer, encoding=args.encoding) if args.input == '-' else smart_open(args.input, encoding=args.encoding)
    full_system = inputfh.readlines()

    # Read references
    full_refs = [[] for x in range(max(len(concat_ref_files[0]), args.num_refs))]
    for ref_files in concat_ref_files:
        for refno, ref_file in enumerate(ref_files):
            for lineno, line in enumerate(smart_open(ref_file, encoding=args.encoding), 1):
                if args.num_refs != 1:
                    splits = line.rstrip().split(sep='\t', maxsplit=args.num_refs-1)
                    if len(splits) != args.num_refs:
                        logging.error('FATAL: line {}: expected {} fields, but found {}.'.format(lineno, args.num_refs, len(splits)))
                        sys.exit(17)
                        for refno, split in enumerate(splits):
                            full_refs[refno].append(split)
                else:
                    full_refs[refno].append(line)

    # Filter sentences according to a given origlang
    system, *refs = _filter_subset([full_system, *full_refs], args.test_set, args.langpair, args.origlang, args.subset)
    if len(system) == 0:
        message = 'Test set %s contains no sentence' % args.test_set
        if args.origlang is not None or args.subset is not None:
            message += ' with'
            message += '' if args.origlang is None else ' origlang=' + args.origlang
            message += '' if args.subset is None else ' subset=' + args.subset
        logging.error(message)
        exit(1)

    # Handle sentence level and quit
    if args.sentence_level:
        for output, *references in zip(system, *refs):
            results = []
            for metric in args.metrics:
                if metric == 'bleu':
                    bleu = sentence_bleu(output,
                                         [[x] for x in references],
                                         smooth_method=args.smooth,
                                         smooth_value=args.smooth_value)
                    results.append(bleu)
                if metric == 'chrf':
                    chrf = sentence_chrf(output,
                                         references[0],
                                         args.chrf_order,
                                         args.chrf_beta,
                                         remove_whitespace=not args.chrf_whitespace)
                    results.append(chrf)

            display_metric(args.metrics, results, len(refs), args)

        sys.exit(0)

    # Else, handle system level
    results = []
    try:
        for metric in args.metrics:
            if metric == 'bleu':
                bleu = corpus_bleu(system, refs, smooth_method=args.smooth, smooth_value=args.smooth_value, force=args.force, lowercase=args.lc, tokenize=args.tokenize)
                results.append(bleu)
            elif metric == 'chrf':
                chrf = corpus_chrf(system, refs[0], beta=args.chrf_beta, order=args.chrf_order, remove_whitespace=not args.chrf_whitespace)
                results.append(chrf)
    except EOFError:
        logging.error('The input and reference stream(s) were of different lengths.')
        if args.test_set is not None:
            logging.error('\nThis could be a problem with your system output or with sacreBLEU\'s reference database.\n'
                          'If the latter, you can clean out the references cache by typing:\n'
                          '\n'
                          '    rm -r %s/%s\n'
                          '\n'
                          'They will be downloaded automatically again the next time you run sacreBLEU.', SACREBLEU_DIR,
                          args.test_set)
        sys.exit(1)

    display_metric(args.metrics, results, len(refs), args)

    if args.detail:
        width = args.width
        sents_digits = len(str(len(full_system)))
        origlangs = args.origlang if args.origlang else _available_origlangs(args.test_set, args.langpair)
        for origlang in origlangs:
            subsets = [None]
            if args.subset is not None:
                subsets += [args.subset]
            elif all(t in SUBSETS for t in args.test_set.split(',')):
                subsets += COUNTRIES + DOMAINS
            for subset in subsets:
                system, *refs = _filter_subset([full_system, *full_refs], args.test_set, args.langpair, origlang, subset)
                if len(system) == 0:
                    continue
                if subset in COUNTRIES:
                    subset_str = '%20s' % ('country=' + subset)
                elif subset in DOMAINS:
                    subset_str = '%20s' % ('domain=' + subset)
                else:
                    subset_str = '%20s' % ''
                if 'bleu' in args.metrics:
                    bleu = corpus_bleu(system, refs, smooth_method=args.smooth, smooth_value=args.smooth_value, force=args.force, lowercase=args.lc, tokenize=args.tokenize)
                    print('origlang={} {}: sentences={:{}} BLEU={:{}.{}f}'.format(origlang, subset_str, len(system), sents_digits, bleu.score, width+4, width))
                if 'chrf' in args.metrics:
                    chrf = corpus_chrf(system, refs[0], beta=args.chrf_beta, order=args.chrf_order, remove_whitespace=not args.chrf_whitespace)
                    print('origlang={} {}: sentences={:{}} chrF={:{}.{}f}'.format(origlang, subset_str, len(system), sents_digits, chrf.score, width+4, width))


def display_metric(metrics_to_print, results, num_refs, args):
    """
    Badly in need of refactoring.
    One idea is to put all of this in the BLEU and CHRF classes, and then define
    a Result::signature() function.
    """
    for metric, result in zip(metrics_to_print, results):
        if metric == 'bleu':
            if args.score_only:
                print('{0:.{1}f}'.format(result.score, args.width))
            else:
                version_str = bleu_signature(args, num_refs)
                print(result.format(args.width).replace('BLEU', 'BLEU+' + version_str))

        elif metric == 'chrf':
            if args.score_only:
                print('{0:.{1}f}'.format(result.score, args.width))
            else:
                version_str = chrf_signature(args, num_refs)
                print('chrF{0:d}+{1} = {2:.{3}f}'.format(args.chrf_beta, version_str, result.score, args.width))


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description='sacreBLEU: Hassle-free computation of shareable BLEU scores.\n'
                    'Quick usage: score your detokenized output against WMT\'14 EN-DE:\n'
                    '    cat output.detok.de | sacrebleu -t wmt14 -l en-de',
        # epilog = 'Available test sets: ' + ','.join(sorted(DATASETS.keys(), reverse=True)),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    arg_parser.add_argument('--test-set', '-t', type=str, default=None,
                            help='the test set to use (see also --list) or a comma-separated list of test sets to be concatenated')
    arg_parser.add_argument('-lc', action='store_true', default=False,
                            help='Use case-insensitive BLEU (default: actual case)')
    arg_parser.add_argument('--sentence-level', '-sl', action='store_true',
                            help='Output metric on each sentence.')
    arg_parser.add_argument('--smooth', '-s', choices=['exp', 'floor', 'add-k', 'none'],
                            default='exp',
                            help='smoothing method: exponential decay (default), floor (increment zero counts), add-k (increment num/denom by k for n>1), or none')
    arg_parser.add_argument('--smooth-value', '-sv', type=float, default=None,
                            help='The value to pass to the smoothing technique, only used for floor and add-k. Default floor: {}, add-k: {}.'.format(
                                SMOOTH_VALUE_DEFAULT['floor'], SMOOTH_VALUE_DEFAULT['add-k']))
    arg_parser.add_argument('--tokenize', '-tok', choices=TOKENIZERS.keys(), default=None,
                            help='tokenization method to use')
    arg_parser.add_argument('--language-pair', '-l', dest='langpair', default=None,
                            help='source-target language pair (2-char ISO639-1 codes)')
    arg_parser.add_argument('--origlang', '-ol', dest='origlang', default=None,
                            help='use a subset of sentences with a given original language (2-char ISO639-1 codes), "non-" prefix means negation')
    arg_parser.add_argument('--subset', dest='subset', default=None,
                            help='use a subset of sentences whose document annotation matches a give regex (see SUBSETS in the source code)')
    arg_parser.add_argument('--download', type=str, default=None,
                            help='download a test set and quit')
    arg_parser.add_argument('--echo', choices=['src', 'ref', 'both'], type=str, default=None,
                            help='output the source (src), reference (ref), or both (both, pasted) to STDOUT and quit')
    arg_parser.add_argument('--input', '-i', type=str, default='-',
                            help='Read input from a file instead of STDIN')
    arg_parser.add_argument('--num-refs', '-nr', type=int, default=1,
                            help='Split the reference stream on tabs, and expect this many references. Default: %(default)s.')
    arg_parser.add_argument('refs', nargs='*', default=[],
                            help='optional list of references (for backwards-compatibility with older scripts)')
    arg_parser.add_argument('--metrics', '-m', choices=['bleu', 'chrf'], nargs='+',
                            default=['bleu'],
                            help='metrics to compute (default: bleu)')
    arg_parser.add_argument('--chrf-order', type=int, default=CHRF_ORDER,
                            help='chrf character order (default: %(default)s)')
    arg_parser.add_argument('--chrf-beta', type=int, default=CHRF_BETA,
                            help='chrf BETA parameter (default: %(default)s)')
    arg_parser.add_argument('--chrf-whitespace', action='store_true', default=False,
                            help='include whitespace in chrF calculation (default: %(default)s)')
    arg_parser.add_argument('--short', default=False, action='store_true',
                            help='produce a shorter (less human readable) signature')
    arg_parser.add_argument('--score-only', '-b', default=False, action='store_true',
                            help='output only the BLEU score')
    arg_parser.add_argument('--force', default=False, action='store_true',
                            help='insist that your tokenized input is actually detokenized')
    arg_parser.add_argument('--quiet', '-q', default=False, action='store_true',
                            help='suppress informative output')
    arg_parser.add_argument('--encoding', '-e', type=str, default='utf-8',
                            help='open text files with specified encoding (default: %(default)s)')
    arg_parser.add_argument('--list', default=False, action='store_true',
                            help='print a list of all available test sets.')
    arg_parser.add_argument('--citation', '--cite', default=False, action='store_true',
                            help='dump the bibtex citation and quit.')
    arg_parser.add_argument('--width', '-w', type=int, default=1,
                            help='floating point width (default: %(default)s)')
    arg_parser.add_argument('--detail', '-d', default=False, action='store_true',
                            help='print extra information (split test sets based on origlang)')
    arg_parser.add_argument('-V', '--version', action='version',
                            version='%(prog)s {}'.format(VERSION))
    args = arg_parser.parse_args()
    return args


if __name__ == '__main__':
    main()
