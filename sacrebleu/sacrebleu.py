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

from collections import defaultdict


# Allows calling the script as a standalone utility
# See: https://github.com/mjpost/sacrebleu/issues/86
if __package__ is None and __name__ == '__main__':
    parent = pathlib.Path(__file__).absolute().parents[1]
    sys.path.insert(0, str(parent))
    __package__ = 'sacrebleu'

from .tokenizers import BLEU_TOKENIZERS
from .dataset import DATASETS
from .metrics import METRICS
from .utils import smart_open, filter_subset, get_langpairs_for_testset, get_available_testsets
from .utils import print_test_set, print_subset_results, get_reference_files, download_test_set
from .utils import args_to_dict, sanity_check_lengths, get_results_table
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
                            help='Dump the bibtex citation and quit.')
    arg_parser.add_argument('--list', default=False, action='store_true',
                            help='Print a list of all available test sets.')
    arg_parser.add_argument('--test-set', '-t', type=str, default=None,
                            help='The test set to use (see also --list) or a comma-separated list of test sets to be concatenated')
    arg_parser.add_argument('--language-pair', '-l', dest='langpair', default=None,
                            help='Source-target language pair (2-char ISO639-1 codes)')
    arg_parser.add_argument('--origlang', '-ol', dest='origlang', default=None,
                            help='Use a subset of sentences with a given original language (2-char ISO639-1 codes), "non-" prefix means negation')
    arg_parser.add_argument('--subset', dest='subset', default=None,
                            help='Use a subset of sentences whose document annotation matches a give regex (see SUBSETS in the source code)')
    arg_parser.add_argument('--download', type=str, default=None,
                            help='Download a test set and quit')
    arg_parser.add_argument('--echo', choices=['src', 'ref', 'both'], type=str, default=None,
                            help='Output the source (src), reference (ref), or both (both, pasted) to STDOUT and quit')
    arg_parser.add_argument('-V', '--version', action='version', version=f'%(prog)s {VERSION}')
    arg_parser.add_argument('-v', '--verbose', action='store_true',
                            help='Prints more information to the logger.')

    # I/O related arguments
    # Multiple input files can be provided for significance testing for example
    arg_parser.add_argument('--input', '-i', type=str, nargs='*', default=None,
                            help='Read input from file(s) instead of STDIN.')
    arg_parser.add_argument('refs', nargs='*', default=[],
                            help='Optional list of references. If given, it should preceed the -i/--input argument.')
    arg_parser.add_argument('--num-refs', '-nr', type=int, default=1,
                            help='Split the reference stream on tabs, and expect this many references. (Default: %(default)s)')
    arg_parser.add_argument('--encoding', '-e', type=str, default='utf-8',
                            help='Open text files with specified encoding (Default: %(default)s)')

    # Significance testing options
    sign_args = arg_parser.add_argument_group('Significance testing related arguments')
    sign_args.add_argument('--bootstrap', '-bs', action='store_true',
                           help='Enable bootstrap resampling for population score estimates.')
    sign_args.add_argument('--paired-bootstrap', '-pbs', action='store_true',
                           help='Compare multiple systems with paired bootstrap resampling.')
    sign_args.add_argument('--n-bootstrap', '-nb', type=int, default=1000,
                           help='Number of bootstrap samples (Default: %(default)s)')

    # Metric selection
    arg_parser.add_argument('--metrics', '-m', choices=METRICS.keys(), nargs='+', default=['bleu'],
                            help='Space-delimited list of metrics to compute (Default: bleu)')
    arg_parser.add_argument('--sentence-level', '-sl', action='store_true', help='Compute the metric for each sentence.')

    # BLEU-related arguments
    # since sacreBLEU had only support for BLEU initially, the argument names
    # are not prefixed with 'bleu' as in chrF arguments for example.
    # Let's do that manually here through dest= options, as otherwise
    # things will get quite hard to maintain when other metrics are added.
    bleu_args = arg_parser.add_argument_group('BLEU related arguments')

    bleu_args.add_argument('--smooth-method', '-s', choices=METRICS['bleu'].SMOOTH_DEFAULTS.keys(), default='exp',
                           dest='bleu_smooth_method',
                           help='Smoothing method: exponential decay, floor (increment zero counts), add-k (increment num/denom by k for n>1), or none. (Default: %(default)s)')
    bleu_args.add_argument('--smooth-value', '-sv', type=float, default=None,
                           dest='bleu_smooth_value',
                           help='The smoothing value. Only valid for floor and add-k. '
                                f"(Defaults: floor: {METRICS['bleu'].SMOOTH_DEFAULTS['floor']}, "
                                f"add-k: {METRICS['bleu'].SMOOTH_DEFAULTS['add-k']})")
    bleu_args.add_argument('--tokenize', '-tok', choices=BLEU_TOKENIZERS.keys(), default=None,
                           dest='bleu_tokenize',
                           help='Tokenization method to use for BLEU. If not provided, defaults to `zh` for Chinese, `ja-mecab` for Japanese and `13a` (mteval) otherwise.')
    arg_parser.add_argument('-lc', dest='bleu_lowercase', action='store_true', default=False,
                            help='If True, enables case-insensitivity. (Default: %(default)s)')
    bleu_args.add_argument('--force', default=False, action='store_true',
                           dest='bleu_force',
                           help='Insist that your tokenized input is actually detokenized')

    # ChrF-related arguments
    chrf_args = arg_parser.add_argument_group('chrF++ related arguments')
    chrf_args.add_argument('--chrf-char-order', type=int, default=METRICS['chrf'].CHAR_ORDER,
                           help='chrF++ character order (Default: %(default)s)')
    chrf_args.add_argument('--chrf-word-order', type=int, default=METRICS['chrf'].WORD_ORDER,
                           help='chrF++ word order (Default: %(default)s)')
    chrf_args.add_argument('--chrf-beta', type=int, default=METRICS['chrf'].BETA,
                           help='chrF BETA parameter (Default: %(default)s)')
    chrf_args.add_argument('--chrf-whitespace', action='store_true', default=False,
                           help='Include whitespace in chrF calculation (Default: %(default)s)')
    chrf_args.add_argument('--chrf-lowercase', action='store_true', default=False,
                           help='If True, enables case-insensitivity. (Default: %(default)s)')

    # TER related arguments
    ter_args = arg_parser.add_argument_group('TER related arguments')
    ter_args.add_argument('--ter-case-sensitive', action='store_true',
                          help='Enables case sensitivity (Default: %(default)s)')
    ter_args.add_argument('--ter-asian-support', action='store_true',
                          help='Enables special treatment of Asian characters (Default: %(default)s)')
    ter_args.add_argument('--ter-no-punct', action='store_true',
                          help='Removes punctuation. (Default: %(default)s)')
    ter_args.add_argument('--ter-normalized', action='store_true',
                          help='Applies basic normalization and tokenization. (Default: %(default)s)')

    # Reporting related arguments
    report_args = arg_parser.add_argument_group('Reporting related arguments')
    report_args.add_argument('--quiet', '-q', default=False, action='store_true',
                             help='Suppress informative output')
    report_args.add_argument('--short', default=False, action='store_true',
                             help='Produce a shorter (less human readable) signature')
    report_args.add_argument('--score-only', '-b', default=False, action='store_true',
                             help='Output only the computed score')
    report_args.add_argument('--width', '-w', type=int, default=1,
                             help='Floating point width (Default: %(default)s)')
    report_args.add_argument('-x', '--latex', action='store_true',
                             help='Produce a LaTeX table in multi-system evaluation mode.')
    report_args.add_argument('--detail', '-d', default=False, action='store_true',
                             help='Print extra information (split test sets based on origlang)')

    args = arg_parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.verbose:
        sacrelogger.setLevel(logging.DEBUG)

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
            for testset in sorted(get_available_testsets()):
                desc = DATASETS[testset].get('description', '').strip()
                print(f'{testset:<30}: {desc}')
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
                sacrelogger.error(f'No citation found for {test_set}')
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
                sacrelogger.error(f'Unknown test set {test_set!r}')
                sacrelogger.error('Please run with --list to see the available test sets.')
                sys.exit(1)

    if args.test_set is None:
        if len(args.refs) == 0:
            sacrelogger.error('If manual references given, make sure to provide them '
                              'before the -i/--input argument to avoid confusion.')
            sacrelogger.error('Otherwise, I need a predefined test set (-t) from the following list:')
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
                sacrelogger.error(f'No such language pair {args.langpair!r}')
                sacrelogger.error(f'Available language pairs for test set {test_set!r} are:')
                for lp in langpairs:
                    sacrelogger.error(f' > {lp}')
                sys.exit(1)

    if args.echo:
        if args.langpair is None or args.test_set is None:
            sacrelogger.warning("--echo requires a test set (--t) and a language pair (-l)")
            sys.exit(1)
        for test_set in args.test_set.split(','):
            print_test_set(test_set, args.langpair, args.echo, args.origlang, args.subset)
        sys.exit(0)

    # Hack: inject target language info for BLEU, so that it can
    # select the tokenizer based on it
    if args.langpair:
        args.bleu_trg_lang = args.langpair.split('-')[1]

    # concat_ref_files is a list of list of reference filenames
    # (concatenation happens if multiple test sets are given through -t)
    # Example: [[testset1_refA, testset1_refB], [testset2_refA, testset2_refB]]
    concat_ref_files = []
    if args.test_set is None:
        concat_ref_files.append(args.refs)
    else:
        # Multiple test sets can be given
        for test_set in args.test_set.split(','):
            ref_files = get_reference_files(test_set, args.langpair)
            if len(ref_files) == 0:
                sacrelogger.warning(
                    f'No references found for test set {test_set}/{args.langpair}.')
            concat_ref_files.append(ref_files)

    #################
    # Read references
    #################
    full_refs = [[] for x in range(max(len(concat_ref_files[0]), args.num_refs))]
    for ref_files in concat_ref_files:
        for refno, ref_file in enumerate(ref_files):
            for lineno, line in enumerate(smart_open(ref_file, encoding=args.encoding), 1):
                line = line.rstrip()
                if args.num_refs == 1:
                    full_refs[refno].append(line)
                else:
                    refs = line.split(sep='\t', maxsplit=args.num_refs - 1)
                    # NOTE: Could be relaxed for variable number of references
                    if len(refs) != args.num_refs:
                        sacrelogger.error(f'FATAL: line {lineno}: expected {args.num_refs} fields, but found {len(refs)}.')
                        sys.exit(17)
                    for refno, ref in enumerate(refs):
                        full_refs[refno].append(ref)

    # Decide on the number of final references, override the argument
    args.num_refs = len(full_refs)

    # Read hypotheses
    # Can't tokenize yet as each metric has its own way of tokenizing things
    full_systems, sys_names = [], []

    if args.input is None:
        # Read from STDIN
        inputfh = io.TextIOWrapper(sys.stdin.buffer, encoding=args.encoding)

        # guess the number of systems by looking at the first line
        fields = inputfh.readline().rstrip().split('\t')
        num_sys = len(fields)

        # place the first lines already
        full_systems = [[s] for s in fields]

        # No explicit system name, just enumerate
        sys_names = [f'sys{i + 1}' for i in range(num_sys)]

        # Read the rest
        for line in inputfh:
            fields = line.rstrip().split('\t')
            if len(fields) != num_sys:
                sacrelogger.error('FATAL: the number of tab-delimited fields in the input stream differ across lines.')
                sys.exit(17)
            # Place systems into the list
            for sys_idx, sent in enumerate(fields):
                full_systems[sys_idx].append(sent.rstrip())
    else:
        # Separate files are given for each system output
        # Ex: --input smt.txt nmt.txt
        num_sys = len(args.input)

        # systems is a list of list of sentences
        for fname in args.input:
            # Base name is assumed to be system ID, normalize underscores for LaTeX
            sys_name = pathlib.Path(fname).name.replace('_', '-')
            sacrelogger.debug(f'{fname!r} will be named as {sys_name!r}')
            if sys_name in sys_names:
                sacrelogger.error(f"{sys_name!r} already used to name a system, modify your paths.")
                sys.exit(1)

            sys_names.append(sys_name)
            inputfh = smart_open(fname, encoding=args.encoding)
            lines = []
            for line in inputfh:
                lines.append(line.rstrip())
            full_systems.append(lines)

    if num_sys > 1:
        if args.sentence_level:
            sacrelogger.error('Only one system can be evaluated in sentence-level mode.')
            sys.exit(1)

        sacrelogger.debug(f"Detected {num_sys} different systems at input.")

    # Filter subsets if requested
    outputs = filter_subset(
        [*full_systems, *full_refs], args.test_set, args.langpair,
        args.origlang, args.subset)

    # Unpack systems & references back
    systems, refs = outputs[:num_sys], outputs[num_sys:]

    # Perform some sanity checks
    for system in systems:
        if len(system) == 0:
            message = f'Test set {args.test_set!r} contains no sentence'
            if args.origlang is not None or args.subset is not None:
                message += ' with'
                if args.origlang:
                    message += f' origlang={args.origlang}'
                if args.subset:
                    message += f' subset={args.subset}' + args.subset
            sacrelogger.error(message)
            sys.exit(1)

        # Check lengths
        sanity_check_lengths(system, refs, test_set=args.test_set)

    if num_sys == 1 and args.paired_bootstrap:
        sacrebleu.error(
            'Paired bootstrap resampling requires multiple input systems')
        sacrebleu.error('For single system estimates, use --bootstrap')
        sys.exit(1)

    if not args.bootstrap and not args.paired_bootstrap:
        # Reset to 1. This is to have a default of 1000 in argparse' defaults
        args.n_bootstrap = 1

    # Create the metrics
    metrics = {}
    for name in args.metrics:
        # Each metric's specific arguments are prefixed with `metricname_`
        # for grouping. Filter accordingly and strip the prefixes prior to
        # metric object construction.
        name = name.lower()
        metric_args = args_to_dict(args, name, strip_prefix=True)
        metrics[name] = METRICS[name](**metric_args)

    # Handle sentence level and quit
    if args.sentence_level:
        # one metric and one system in use for sentence-level
        metric, system = metrics[args.metrics[0]], systems[0]

        for output, *references in zip(system, *refs):
            score = metric.sentence_score(output, references)
            sig = metric.signature.get(args.short)
            print(score.format(args.width, args.score_only, sig))

        sys.exit(0)

    # Corpus level evaluation mode
    if num_sys == 1:
        for name in sorted(metrics):
            # Get the signature
            score = metrics[name].corpus_score(
                system, refs, n_bootstrap=args.n_bootstrap)
            sig = metrics[name].signature.get(short=args.short)
            print(score.format(args.width, args.score_only, sig))

    else:
        sigs = {}
        scores = defaultdict(list)
        scores['System'] = sys_names

        for idx, system in enumerate(systems):
            sys_name = sys_names[idx]
            sacrelogger.debug(f'Evaluating {sys_name!r}')
            for name in sorted(metrics):
                score = metrics[name].corpus_score(
                    system, refs, n_bootstrap=args.n_bootstrap)
                sigs[name] = metrics[name].signature.get(args.short)
                scores[score.prefix].append(score.format(args.width, True))

        print(get_results_table(scores, latex=args.latex))
        print()
        print('Metric signatures:')
        for name, sig in sigs.items():
            print(f' {name:<10} {sig}')

    if args.bootstrap or args.paired_bootstrap:
        print()
        if args.bootstrap:
            print(f'Bootstrap resampling (n={args.n_bootstrap}) done to '
                  'provide 95% confidence intervals around the true mean.')
        if args.paired_bootstrap:
            print(f'Paired bootstrap resampling (n={args.n_bootstrap}) done to '
                  'compare candidate systems against a baseline.')

    # Prints detailed information for translationese effect experiments
    # FIXME: What happens with many systems here
    if args.detail:
        if num_sys == 1:
            print_subset_results(metrics, full_systems[0], full_refs, args)


if __name__ == '__main__':
    main()
