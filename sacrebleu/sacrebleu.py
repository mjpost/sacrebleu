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

import io
import sys
import logging
import pathlib
import argparse

# Allows calling the script as a standalone utility
# See: https://github.com/mjpost/sacrebleu/issues/86
if __package__ is None and __name__ == '__main__':
    parent = pathlib.Path(__file__).absolute().parents[1]
    sys.path.insert(0, str(parent))
    __package__ = 'sacrebleu'

from .tokenizers import TOKENIZERS, DEFAULT_TOKENIZER
from .dataset import DATASETS, DOMAINS, COUNTRIES, SUBSETS
from .metrics import METRICS

from .utils import smart_open, filter_subset, get_available_origlangs, SACREBLEU_DIR
from .utils import get_langpairs_for_testset, get_available_testsets
from .utils import print_test_set, get_reference_files, download_test_set
from .utils import args_to_dict
from . import __version__ as VERSION

sacrelogger = logging.getLogger('sacrebleu')

try:
    # SIGPIPE is not available on Windows machines, throwing an exception.
    from signal import SIGPIPE

    # If SIGPIPE is available, change behaviour to default instead of ignore.
    from signal import signal, SIG_DFL
    signal(SIGPIPE, SIG_DFL)

except ImportError:
    sacrelogger.warning('Could not import signal.SIGPIPE (this is expected on Windows machines)')


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description='sacreBLEU: Hassle-free computation of shareable BLEU scores.\n'
                    'Quick usage: score your detokenized output against WMT\'14 EN-DE:\n'
                    '    cat output.detok.de | sacrebleu -t wmt14 -l en-de',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    arg_parser.add_argument('--citation', '--cite', default=False, action='store_true',
                            help='dump the bibtex citation and quit.')
    arg_parser.add_argument('--list', default=False, action='store_true',
                            help='print a list of all available test sets.')
    arg_parser.add_argument('--test-set', '-t', type=str, default=None,
                            help='the test set to use (see also --list) or a comma-separated list of test sets to be concatenated')
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
    arg_parser.add_argument('-V', '--version', action='version',
                            version='%(prog)s {}'.format(VERSION))

    # I/O related arguments
    arg_parser.add_argument('--input', '-i', type=str, default='-',
                            help='Read input from a file instead of STDIN')
    arg_parser.add_argument('refs', nargs='*', default=[],
                            help='optional list of references (for backwards-compatibility with older scripts)')
    arg_parser.add_argument('--num-refs', '-nr', type=int, default=1,
                            help='Split the reference stream on tabs, and expect this many references. Default: %(default)s.')
    arg_parser.add_argument('--encoding', '-e', type=str, default='utf-8',
                            help='open text files with specified encoding (default: %(default)s)')

    # Metric selection
    arg_parser.add_argument('--metrics', '-m', choices=METRICS.keys(), nargs='+', default=['bleu'],
                            help='metrics to compute (default: bleu)')
    arg_parser.add_argument('--sentence-level', '-sl', action='store_true', help='Compute the metric for each sentence.')

    # BLEU-related arguments
    # since sacreBLEU had only support for BLEU initially, the argument names
    # are not prefixed with 'bleu' as in chrF arguments for example.
    # Let's do that manually here through dest= options, as otherwise
    # things will get quite hard to maintain when other metrics are added.
    bleu_args = arg_parser.add_argument_group('BLEU related arguments')
    bleu_args.add_argument('-lc', action='store_true', default=False,
                           dest='bleu_lc', help='Use case-insensitive BLEU (default: False)')
    bleu_args.add_argument('--smooth-method', '-s', choices=METRICS['bleu'].SMOOTH_DEFAULTS.keys(), default='exp',
                           dest='bleu_smooth_method',
                           help='smoothing method: exponential decay (default), floor (increment zero counts), add-k (increment num/denom by k for n>1), or none')
    bleu_args.add_argument('--smooth-value', '-sv', type=float, default=None,
                           dest='bleu_smooth_value',
                           help='The smoothing value. Only valid for floor and add-k. (Defaults: floor: {}, add-k: {})'.format(
                               METRICS['bleu'].SMOOTH_DEFAULTS['floor'], METRICS['bleu'].SMOOTH_DEFAULTS['add-k']))
    bleu_args.add_argument('--tokenize', '-tok', choices=TOKENIZERS.keys(), default=None,
                           dest='bleu_tokenize',
                           help='Tokenization method to use for BLEU. If not provided, defaults to `zh` for Chinese, `mecab` for Japanese and `v13a` (mteval) otherwise.')
    bleu_args.add_argument('--force', default=False, action='store_true',
                           dest='bleu_force',
                           help='insist that your tokenized input is actually detokenized')

    # ChrF-related arguments
    chrf_args = arg_parser.add_argument_group('chrF related arguments')
    chrf_args.add_argument('--chrf-order', type=int, default=METRICS['chrf'].ORDER,
                           help='chrf character order (default: %(default)s)')
    chrf_args.add_argument('--chrf-beta', type=int, default=METRICS['chrf'].BETA,
                           help='chrf BETA parameter (default: %(default)s)')
    chrf_args.add_argument('--chrf-whitespace', action='store_true', default=False,
                           help='include whitespace in chrF calculation (default: %(default)s)')

    # Reporting related arguments
    report_args = arg_parser.add_argument_group('Reporting related arguments')
    report_args.add_argument('--quiet', '-q', default=False, action='store_true',
                             help='suppress informative output')
    report_args.add_argument('--short', default=False, action='store_true',
                             help='produce a shorter (less human readable) signature')
    report_args.add_argument('--score-only', '-b', default=False, action='store_true',
                             help='output only the BLEU score')
    report_args.add_argument('--width', '-w', type=int, default=1,
                             help='floating point width (default: %(default)s)')
    report_args.add_argument('--detail', '-d', default=False, action='store_true',
                             help='print extra information (split test sets based on origlang)')

    args = arg_parser.parse_args()
    return args


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
            print('The available test sets are:')
            for testset in get_available_testsets():
                print('%30s: %s' % (testset, DATASETS[testset].get('description', '').strip()))
        sys.exit(0)

    if args.sentence_level and len(args.metrics) > 1:
        sacrelogger.error('Only one metric can be used with Sentence-level reporting.')
        sys.exit(1)

    if args.citation:
        if not args.test_set:
            sacrelogger.error('I need a test set (-t).')
            sys.exit(1)
        for test_set in args.test_set.split(','):
            if 'citation' not in DATASETS[test_set]:
                sacrelogger.error('No citation found for %s', test_set)
            else:
                print(DATASETS[test_set]['citation'])
        sys.exit(0)

    if args.num_refs != 1 and (args.test_set is not None or len(args.refs) > 1):
        sacrelogger.error('The --num-refs argument allows you to provide any number of tab-delimited references in a single file.')
        sacrelogger.error('You can only use it with externally provided references, however (i.e., not with `-t`),')
        sacrelogger.error('and you cannot then provide multiple reference files.')
        sys.exit(1)

    if args.test_set is not None:
        for test_set in args.test_set.split(','):
            if test_set not in DATASETS:
                sacrelogger.error('Unknown test set "%s"', test_set)
                sacrelogger.error('Please run with --list to see the available test sets.')
                sys.exit(1)

    if args.test_set is None:
        if len(args.refs) == 0:
            sacrelogger.error('I need either a predefined test set (-t) or a list of references')
            sacrelogger.error(get_available_testsets())
            sys.exit(1)
    elif len(args.refs) > 0:
        sacrelogger.error('I need exactly one of (a) a predefined test set (-t) or (b) a list of references')
        sys.exit(1)
    elif args.langpair is None:
        sacrelogger.error('I need a language pair (-l).')
        sys.exit(1)
    else:
        for test_set in args.test_set.split(','):
            langpairs = get_langpairs_for_testset(test_set)
            if args.langpair not in langpairs:
                sacrelogger.error('No such language pair "%s"', args.langpair)
                sacrelogger.error('Available language pairs for test set "%s": %s', test_set,
                                  ', '.join(langpairs))
                sys.exit(1)

    if args.echo:
        if args.langpair is None or args.test_set is None:
            sacrelogger.warning("--echo requires a test set (--t) and a language pair (-l)")
            sys.exit(1)
        for test_set in args.test_set.split(','):
            print_test_set(test_set, args.langpair, args.echo, args.origlang, args.subset)
        sys.exit(0)

    if args.test_set is not None and args.bleu_tokenize == 'none':
        sacrelogger.warning("You are turning off sacrebleu's internal tokenization ('--tokenize none'), presumably to supply\n"
                            "your own reference tokenization. Published numbers will not be comparable with other papers.\n")

    if 'ter' in args.metrics and args.bleu_tokenize is not None:
        logging.warning("Your setting of --tokenize will be ignored when "
                        "computing TER")

    # Parse languages for convenience
    src_lang, trg_lang = '', ''
    if args.langpair:
        src_lang, trg_lang = args.langpair.split('-')

    # Default tokenizer assignment (only affects BLEU as of now)
    if args.bleu_tokenize is None:
        # set default
        if trg_lang == 'zh':
            args.bleu_tokenize = 'zh'
        elif trg_lang == 'ja':
            args.bleu_tokenize = 'ja-mecab'
        else:
            args.bleu_tokenize = DEFAULT_TOKENIZER

    if args.langpair is not None and 'bleu' in args.metrics:
        if trg_lang == 'zh' and args.bleu_tokenize != 'zh':
            sacrelogger.warning('You should pass "--tok zh" when scoring Chinese.')
        if trg_lang == 'ja' and not args.bleu_tokenize.startswith('ja-'):
            sacrelogger.warning('You should pass "--tok ja-mecab" when scoring Japanese.')

    # concat_ref_files is a list of list of reference filenames, for example:
    # concat_ref_files = [[testset1_refA, testset1_refB], [testset2_refA, testset2_refB]]
    if args.test_set is None:
        concat_ref_files = [args.refs]
    else:
        concat_ref_files = []
        for test_set in args.test_set.split(','):
            ref_files = get_reference_files(test_set, args.langpair)
            if len(ref_files) == 0:
                sacrelogger.warning('No references found for test set {}/{}.'.format(test_set, args.langpair))
            concat_ref_files.append(ref_files)

    # Read references
    full_refs = [[] for x in range(max(len(concat_ref_files[0]), args.num_refs))]
    for ref_files in concat_ref_files:
        for refno, ref_file in enumerate(ref_files):
            for lineno, line in enumerate(smart_open(ref_file, encoding=args.encoding), 1):
                if args.num_refs != 1:
                    splits = line.rstrip().split(sep='\t', maxsplit=args.num_refs-1)
                    if len(splits) != args.num_refs:
                        sacrelogger.error('FATAL: line {}: expected {} fields, but found {}.'.format(lineno, args.num_refs, len(splits)))
                        sys.exit(17)
                    for refno, split in enumerate(splits):
                        full_refs[refno].append(split)
                else:
                    full_refs[refno].append(line)

    # Decide on the number of final references, override the argument
    args.num_refs = len(full_refs)

    # Read hypotheses stream
    if args.input == '-':
        inputfh = io.TextIOWrapper(sys.stdin.buffer, encoding=args.encoding)
    else:
        inputfh = smart_open(args.input, encoding=args.encoding)
    full_system = inputfh.readlines()

    # Filter sentences according to a given origlang
    system, *refs = filter_subset(
        [full_system, *full_refs], args.test_set, args.langpair, args.origlang, args.subset)

    if len(system) == 0:
        message = 'Test set %s contains no sentence' % args.test_set
        if args.origlang is not None or args.subset is not None:
            message += ' with'
            message += '' if args.origlang is None else ' origlang=' + args.origlang
            message += '' if args.subset is None else ' subset=' + args.subset
        sacrelogger.error(message)
        sys.exit(1)

    # Create metric inventory, let each metric consume relevant args from argparse
    metrics = []
    for metric in args.metrics:
        # Each metric's specific arguments are prefixed with `metricname_`
        # for grouping. Filter accordingly and strip the prefixes prior to
        # metric object construction.
        metric_args = args_to_dict(args, metric.lower(), strip_prefix=True)
        metrics.append(METRICS[metric](**metric_args))

    # Handle sentence level and quit
    if args.sentence_level:
        # one metric in use for sentence-level
        metric = metrics[0]
        sig = metric.signature.get(short=args.short)
        for output, *references in zip(system, *refs):
            score = metric.sentence_score(output, references)
            print(score.format(args.width, args.score_only, sig))

        sys.exit(0)

    # Else, handle system level
    for metric in metrics:
        try:
            score = metric.corpus_score(system, refs)
        except EOFError:
            sacrelogger.error('The input and reference stream(s) were of different lengths.')
            if args.test_set is not None:
                sacrelogger.error('\nThis could be a problem with your system output or with sacreBLEU\'s reference database.\n'
                                  'If the latter, you can clean out the references cache by typing:\n'
                                  '\n'
                                  '    rm -r %s/%s\n'
                                  '\n'
                                  'They will be downloaded automatically again the next time you run sacreBLEU.', SACREBLEU_DIR,
                                  args.test_set)
            sys.exit(1)
        else:
            sig = metric.signature.get(short=args.short)
            print(score.format(args.width, args.score_only, sig))

    if args.detail:
        width = args.width
        sents_digits = len(str(len(full_system)))
        origlangs = args.origlang if args.origlang else get_available_origlangs(args.test_set, args.langpair)
        for origlang in origlangs:
            subsets = [None]
            if args.subset is not None:
                subsets += [args.subset]
            elif all(t in SUBSETS for t in args.test_set.split(',')):
                subsets += COUNTRIES + DOMAINS
            for subset in subsets:
                system, *refs = filter_subset([full_system, *full_refs], args.test_set, args.langpair, origlang, subset)
                if len(system) == 0:
                    continue
                if subset in COUNTRIES:
                    subset_str = '%20s' % ('country=' + subset)
                elif subset in DOMAINS:
                    subset_str = '%20s' % ('domain=' + subset)
                else:
                    subset_str = '%20s' % ''
                for metric in metrics:
                    # FIXME: handle this in metrics
                    if metric.name == 'bleu':
                        _refs = refs
                    elif metric.name == 'chrf':
                        _refs = refs[0]

                    score = metric.corpus_score(system, _refs)
                    print('origlang={} {}: sentences={:{}} {}={:{}.{}f}'.format(
                        origlang, subset_str, len(system), sents_digits,
                        score.prefix, score.score, width+4, width))


if __name__ == '__main__':
    main()
