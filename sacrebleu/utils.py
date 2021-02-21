import os
import re
import ssl
import sys
import gzip
import math
import hashlib
import logging
import urllib.request
from itertools import filterfalse
from typing import List, Iterable, Optional

import portalocker
from tabulate import tabulate

from .dataset import DATASETS, SUBSETS, DOMAINS, COUNTRIES

# Where to store downloaded test sets.
# Define the environment variable $SACREBLEU, or use the default of ~/.sacrebleu.
#
# Querying for a HOME environment variable can result in None (e.g., on Windows)
# in which case the os.path.join() throws a TypeError. Using expanduser() is
# a safe way to get the user's home folder.
USERHOME = os.path.expanduser("~")
SACREBLEU_DIR = os.environ.get('SACREBLEU', os.path.join(USERHOME, '.sacrebleu'))

sacrelogger = logging.getLogger('sacrebleu')


def get_results_table(results: dict,
                      latex: bool = False):
    tablefmt = 'latex_booktabs' if latex else 'grid'
    s = tabulate(
        results, headers='keys',
        tablefmt=tablefmt,
        floatfmt='.2f')
    return s


def sanity_check_lengths(system: Iterable[str],
                         refs: List[Iterable[str]],
                         test_set: Optional[str] = None):
    n_hyps = len(system)
    if any(len(ref_stream) != n_hyps for ref_stream in refs):
        sacrelogger.error("System and reference streams have different lengths.")
        if test_set:
            sacrelogger.error("This could be an issue with your system output "
                              "or with sacreBLEU's reference database if -t is given.")
            sacrelogger.error("For the latter, try cleaning out the cache by typing:\n")
            sacrelogger.error(f"  rm -r {SACREBLEU_DIR}/{test_set}\n")
            sacrelogger.error("The test sets will be downloaded again next time you run sacreBLEU.")
        sys.exit(1)


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


def args_to_dict(args, prefix, strip_prefix: bool = False):
    """Converts argparse's `Namespace` into dictionary with only arguments
    beginning with the given prefix."""
    prefix += '_'
    d = {}
    for k, v in args.__dict__.items():
        if k.startswith(prefix):
            k = k.replace(prefix, '') if strip_prefix else k
            d[k] = v
    # add global lowercase and num_refs arguments as well
    d['num_refs'] = args.num_refs
    d['lowercase'] = args.lc
    return d


def process_to_text(rawfile, txtfile, field: int = None):
    """Processes raw files to plain text files. Can handle SGML, XML, TSV files, and plain text.
    Called after downloading datasets.

    :param rawfile: the input file (possibly SGML)
    :param txtfile: the plaintext file
    :param field: For TSV files, which field to extract.
    """
    def _clean(s):
        """
        Removes trailing and leading spaces and collapses multiple consecutive internal spaces to a single one.

        :param s: The string.
        :return: A cleaned-up string.
        """
        return re.sub(r'\s+', ' ', s.strip())

    if not os.path.exists(txtfile) or os.path.getsize(txtfile) == 0:
        sacrelogger.info(f"Processing {rawfile} to {txtfile}")
        if rawfile.endswith('.sgm') or rawfile.endswith('.sgml'):
            with smart_open(rawfile) as fin, smart_open(txtfile, 'wt') as fout:
                for line in fin:
                    if line.startswith('<seg '):
                        print(_clean(re.sub(r'<seg.*?>(.*)</seg>.*?', '\\1', line)), file=fout)
        # IWSLT
        elif rawfile.endswith('.xml'):
            with smart_open(rawfile) as fin, smart_open(txtfile, 'wt') as fout:
                for line in fin:
                    if line.startswith('<seg '):
                        print(_clean(re.sub(r'<seg.*?>(.*)</seg>.*?', '\\1', line)), file=fout)
        # MTNT
        elif rawfile.endswith('.tsv'):
            with smart_open(rawfile) as fin, smart_open(txtfile, 'wt') as fout:
                for line in fin:
                    print(line.rstrip().split('\t')[field], file=fout)
        # PLAIN TEXT
        else:
            with smart_open(rawfile) as fin, smart_open(txtfile, 'wt') as fout:
                for line in fin:
                    print(line.rstrip(), file=fout)


def print_test_set(test_set, langpair, side, origlang=None, subset=None):
    """Prints to STDOUT the specified side of the specified test set.

    :param test_set: the test set to print
    :param langpair: the language pair
    :param side: 'src' for source, 'ref' for reference
    :param origlang: print only sentences with a given original language (2-char ISO639-1 code), "non-" prefix means negation
    :param subset: print only sentences whose document annotation matches a given regex
    """
    if side == 'src':
        files = [get_source_file(test_set, langpair)]
    elif side == 'ref':
        files = get_reference_files(test_set, langpair)
    elif side == "both":
        files = [get_source_file(test_set, langpair)] + get_reference_files(test_set, langpair)

    streams = [smart_open(file) for file in files]
    streams = filter_subset(streams, test_set, langpair, origlang, subset)
    for lines in zip(*streams):
        print('\t'.join(map(lambda x: x.rstrip(), lines)))


def get_source_file(test_set, langpair):
    """
    Returns the source file for a given testset/langpair.
    Downloads it first if it is not already local.

    :param test_set: The test set (e.g., "wmt19")
    :param langpair: The language pair (e.g., "de-en")
    :return: the path to the requested source file
    """
    return get_files(test_set, langpair)[0]


def get_reference_files(test_set, langpair):
    """
    Returns a list of one or more reference file paths for the given testset/langpair.
    Downloads the references first if they are not already local.

    :param test_set: The test set (e.g., "wmt19")
    :param langpair: The language pair (e.g., "de-en")
    :return: a list of one or more reference file paths
    """
    return get_files(test_set, langpair)[1:]


def get_files(test_set, langpair):
    """
    Returns the path of the source file and all reference files for
    the provided test set / language pair.
    Downloads the references first if they are not already local.

    :param test_set: The test set (e.g., "wmt19")
    :param langpair: The language pair (e.g., "de-en")
    :return: a list of the source file and all reference files
    """

    if test_set not in DATASETS:
        raise Exception(f"No such test set {test_set}")
    if langpair not in DATASETS[test_set]:
        raise Exception(f"No such language pair {test_set}/{langpair}")

    cachedir = os.path.join(SACREBLEU_DIR, test_set)
    source, target = langpair.split("-")

    source_path = os.path.join(cachedir, f"{langpair}.{source}")

    num_refs = len(DATASETS[test_set][langpair]) - 1
    if num_refs == 1:
        reference_paths = [os.path.join(cachedir, f"{langpair}.{target}")]
    else:
        reference_paths = [os.path.join(cachedir, f"{langpair}.{target}.{num}") for num in range(num_refs)]

    if any(filterfalse(os.path.exists, [source_path] + reference_paths)):
        download_test_set(test_set, langpair)

    return [source_path] + reference_paths


def download_test_set(test_set, langpair=None):
    """Downloads the specified test to the system location specified by the SACREBLEU environment variable.

    :param test_set: the test set to download
    :param langpair: the language pair (needed for some datasets)
    :return: the set of processed file names
    """

    if test_set not in DATASETS:
        raise Exception(f"No such test set {test_set}")

    outdir = os.path.join(SACREBLEU_DIR, test_set)
    os.makedirs(outdir, exist_ok=True)

    expected_checksums = DATASETS[test_set].get('md5', [None] * len(DATASETS[test_set]))
    for dataset, expected_md5 in zip(DATASETS[test_set]['data'], expected_checksums):
        tarball = os.path.join(outdir, os.path.basename(dataset))
        rawdir = os.path.join(outdir, 'raw')

        lockfile = f'{tarball}.lock'
        with portalocker.Lock(lockfile, 'w', timeout=60):
            if not os.path.exists(tarball) or os.path.getsize(tarball) == 0:
                sacrelogger.info(f"Downloading {dataset} to {tarball}")
                try:
                    with urllib.request.urlopen(dataset) as f, open(tarball, 'wb') as out:
                        out.write(f.read())
                except ssl.SSLError:
                    sacrelogger.warning('An SSL error was encountered in downloading the files. If you\'re on a Mac, '
                                        'you may need to run the "Install Certificates.command" file located in the '
                                        '"Python 3" folder, often found under /Applications')
                    sys.exit(1)

                # Check md5sum
                if expected_md5 is not None:
                    md5 = hashlib.md5()
                    with open(tarball, 'rb') as infile:
                        for line in infile:
                            md5.update(line)
                    cur_md5 = md5.hexdigest()
                    if cur_md5 != expected_md5:
                        sacrelogger.error(f'Fatal: MD5 sum of downloaded file was incorrect (got {cur_md5}, expected {expected_md5}).')
                        sacrelogger.error(f'Please manually delete {tarball!r} and rerun the command.')
                        sacrelogger.error('If the problem persists, the tarball may have changed, in which case, please contact the SacreBLEU maintainer.')
                        sys.exit(1)
                    else:
                        sacrelogger.info(f'Checksum passed: {cur_md5}')

                # Extract the tarball
                sacrelogger.info(f'Extracting {tarball}')
                if tarball.endswith('.tar.gz') or tarball.endswith('.tgz'):
                    import tarfile
                    with tarfile.open(tarball) as tar:
                        tar.extractall(path=rawdir)
                elif tarball.endswith('.zip'):
                    import zipfile
                    with zipfile.ZipFile(tarball, 'r') as zipfile:
                        zipfile.extractall(path=rawdir)

    file_paths = []

    # Process the files into plain text
    languages = get_langpairs_for_testset(test_set) if langpair is None else [langpair]
    for pair in languages:
        src, tgt = pair.split('-')
        rawfile = DATASETS[test_set][pair][0]
        field = None  # used for TSV files
        if rawfile.endswith('.tsv'):
            field, rawfile = rawfile.split(':', maxsplit=1)
            field = int(field)
        rawpath = os.path.join(rawdir, rawfile)
        outpath = os.path.join(outdir, f'{pair}.{src}')
        process_to_text(rawpath, outpath, field=field)
        file_paths.append(outpath)

        refs = DATASETS[test_set][pair][1:]
        for i, ref in enumerate(refs):
            field = None
            if ref.endswith('.tsv'):
                field, ref = ref.split(':', maxsplit=1)
                field = int(field)
            rawpath = os.path.join(rawdir, ref)
            if len(refs) >= 2:
                outpath = os.path.join(outdir, f'{pair}.{tgt}.{i}')
            else:
                outpath = os.path.join(outdir, f'{pair}.{tgt}')
            process_to_text(rawpath, outpath, field=field)
            file_paths.append(outpath)

    return file_paths


def get_langpairs_for_testset(testset: str) -> List:
    """Return a list of language pairs for a given test set."""
    return list(filter(lambda x: re.match(r'\w\w\-\w\w', x), DATASETS.get(testset, {}).keys()))


def get_available_testsets() -> List:
    """Return a list of available test sets."""
    return sorted(DATASETS.keys(), reverse=True)


def get_available_origlangs(test_sets, langpair):
    """Return a list of origlang values in according to the raw SGM files."""
    if test_sets is None:
        return []

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


def filter_subset(systems, test_sets, langpair, origlang, subset=None):
    """Filter sentences with a given origlang (or subset) according to the raw SGM files."""
    if origlang is None and subset is None:
        return systems
    if test_sets is None or langpair is None:
        raise ValueError('Filtering for --origlang or --subset needs a test (-t) and a language pair (-l).')

    indices_to_keep = []
    for test_set in test_sets.split(','):
        rawfile = os.path.join(SACREBLEU_DIR, test_set, 'raw', DATASETS[test_set][langpair][0])
        if not rawfile.endswith('.sgm'):
            raise Exception(f'--origlang and --subset supports only *.sgm files, not {rawfile!r}')
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
    return [[sentence for sentence, keep in zip(sys, indices_to_keep) if keep] for sys in systems]


def print_subset_results(metrics, full_system, full_refs, args):
    w = args.width
    ws = len(str(len(full_system)))
    origlangs = args.origlang if args.origlang else \
        get_available_origlangs(args.test_set, args.langpair)

    for origlang in origlangs:
        subsets = [None]
        if args.subset is not None:
            subsets += [args.subset]
        elif all(t in SUBSETS for t in args.test_set.split(',')):
            subsets += COUNTRIES + DOMAINS
        for subset in subsets:
            system, *refs = filter_subset(
                [full_system, *full_refs], args.test_set, args.langpair, origlang, subset)

            if len(system) == 0:
                continue

            subset_str = ''
            if subset in COUNTRIES:
                subset_str = f'country={subset}'
            elif subset in DOMAINS:
                subset_str = f'domain={subset}'

            # pad
            subset_str = f'{subset_str:>20}'

            for metric in metrics.values():
                score = metric.corpus_score(system, refs)
                res = f'origlang={origlang} {subset_str}: sentences={len(system):>{ws}}'
                res += f' {score.prefix}={score.score:{w+4}.{w}f}'
                print(res)
