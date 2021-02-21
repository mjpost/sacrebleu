import math
import logging
from collections import defaultdict
from typing import List, Iterable, Optional, Tuple

from ..tokenizers import TOKENIZERS
from ..utils import my_log
from .base import BaseScore, Signature
from .helpers import extract_word_ngrams

sacrelogger = logging.getLogger('sacrebleu')

# The default for the maximum n-gram order when computing precisions
MAX_NGRAM_ORDER = 4


class BLEUSignature(Signature):
    def __init__(self, args: dict):
        super().__init__(args)

        self._abbr.update({
            'smooth': 's',
            'case': 'c',
            'tok': 'tok',
            'numrefs': '#',
        })

        # Construct a combined string for smoothing method and value
        smooth_str = self.args['smooth_method']
        smooth_def = BLEU.SMOOTH_DEFAULTS[smooth_str]

        # If the method requires a parameter, add it within brackets
        if smooth_def is not None:
            # the following can be None if the user wants to use the default
            smooth_val = self.args['smooth_value']

            if smooth_val is None:
                smooth_val = smooth_def

            smooth_str += '[{smoot-val:.2f}]'

        self.info.update({
            'smooth': smooth_str,
            'case': 'lc' if self.args['lowercase'] else 'mixed',
            'tok': self.args['tokenizer_signature'],
            'numrefs': self.args.get('num_refs', '?'),
        })


class BLEUScore(BaseScore):
    """A convenience class to represent BLEU scores (without signature)."""
    def __init__(self, score, counts, totals, precisions, bp, sys_len, ref_len):
        super().__init__(score)

        self.prefix = 'BLEU'
        self.bp = bp
        self.counts = counts
        self.totals = totals
        self.sys_len = sys_len
        self.ref_len = ref_len
        self.precisions = precisions
        self.prec_str = "/".join([f"{p:.1f}" for p in self.precisions])
        self.ratio = self.sys_len / self.ref_len if self.ref_len else 0

    def format(self, width=2, score_only=False, signature=''):
        if score_only:
            return f'{self.score:.{width}f}'

        pr = f"{self.prefix}+{signature}" if signature else self.prefix
        s = f'{pr} = {self.score:.{width}f} {self.prec_str} '
        s += f'(BP = {self.bp:.3f} ratio = {self.ratio:.3f} '
        s += f'hyp_len = {self.sys_len:d} ref_len = {self.ref_len:d})'
        return s


class BLEU:
    """Computes the BLEU metric given hypotheses and references.

    :param lowercase: If True, lowercased BLEU is computed
    :param force: Ignore data that looks already tokenized
    :param tokenize: The tokenizer to use
    :param smooth_method: The smoothing method to use ('floor', 'add-k', 'exp' or 'none')
    :param smooth_value: The smoothing value for `floor` and `add-k` methods. `None` falls back to default value.
    :param max_ngram_order: If given, it overrides the maximum n-gram order (default: 4) when computing precisions.
    :param num_refs: The number of references given
    """

    SMOOTH_DEFAULTS = {
        # The defaults for `floor` and `add-k` are obtained from the following paper
        # A Systematic Comparison of Smoothing Techniques for Sentence-Level BLEU
        # Boxing Chen and Colin Cherry
        # http://aclweb.org/anthology/W14-3346
        'none': None,   # No value is required
        'floor': 0.1,
        'add-k': 1,
        'exp': None,    # No value is required
    }

    def __init__(self, lowercase: bool = False,
                 force: bool = False,
                 tokenize: str = '13a', smooth_method: str = 'exp',
                 smooth_value: Optional[float] = None,
                 max_ngram_order: int = MAX_NGRAM_ORDER,
                 num_refs: int = 1):
        self.name = 'bleu'
        self.force = force
        self.num_refs = num_refs
        self.lowercase = lowercase
        self.smooth_value = smooth_value
        self.smooth_method = smooth_method
        self.max_ngram_order = max_ngram_order
        self.tokenizer = TOKENIZERS[tokenize]()

        # Sanity check
        assert self.smooth_method in self.SMOOTH_DEFAULTS.keys(), \
            "Unknown smooth_method {self.smooth_method!r}"

        # Build the signature
        self.tokenizer_signature = self.tokenizer.signature()
        self.signature = BLEUSignature(self.__dict__)

    @staticmethod
    def reference_stats(refs, hyp_len, max_ngram_order=MAX_NGRAM_ORDER):
        """Extracts reference statistics for a given segment.

        :param refs: A list of segment tokens.
        :param hyp_len: Hypothesis length for this segment.
        :return: a tuple of (ngrams, closest_ref_len)
        """

        closest_diff = None
        closest_ref_len = None
        ngrams = defaultdict(int)

        for ref in refs:
            # extract n-grams for this ref
            ref_tokens = ref.split()
            ref_len = len(ref_tokens)
            diff = abs(hyp_len - ref_len)
            if closest_diff is None or diff < closest_diff:
                closest_diff = diff
                closest_ref_len = ref_len
            elif diff == closest_diff:
                if ref_len < closest_ref_len:
                    closest_ref_len = ref_len

            cur_ngrams = extract_word_ngrams(ref_tokens, 1, max_ngram_order)
            for key in cur_ngrams.keys():
                ngrams[key] = max(ngrams[key], cur_ngrams[key])

        return dict(ngrams), closest_ref_len

    @staticmethod
    def compute_bleu(correct: List[int],
                     total: List[int],
                     sys_len: int,
                     ref_len: int,
                     smooth_method: str = 'none',
                     smooth_value=None,
                     use_effective_order: bool = False,
                     max_ngram_order: int = MAX_NGRAM_ORDER) -> BLEUScore:
        """Computes BLEU score from its sufficient statistics. Adds smoothing.

        Smoothing methods (citing "A Systematic Comparison of Smoothing Techniques for Sentence-Level BLEU",
        Boxing Chen and Colin Cherry, WMT 2014: http://aclweb.org/anthology/W14-3346)

        - none: No smoothing.
        - floor: Method 1 (requires small positive value (0.1 in the paper) to be set)
        - add-k: Method 2 (Generalizing Lin and Och, 2004)
        - exp: Method 3 (NIST smoothing method i.e. in use with mteval-v13a.pl)

        :param correct: List of counts of correct ngrams, 1 <= n <= max_ngram_order
        :param total: List of counts of total ngrams, 1 <= n <= max_ngram_order
        :param sys_len: The cumulative system length
        :param ref_len: The cumulative reference length
        :param smooth_method: The smoothing method to use ('floor', 'add-k', 'exp' or 'none')
        :param smooth_value: The smoothing value for `floor` and `add-k` methods. `None` falls back to default value.
        :param use_effective_order: If true, use the length of `correct` for the n-gram order instead of max_ngram_order.
        :param max_ngram_order: If given, it overrides the maximum n-gram order (default: 4) when computing precisions.
        :return: A BLEU object with the score (100-based) and other statistics.
        """
        assert smooth_method in BLEU.SMOOTH_DEFAULTS.keys(), \
            "Unknown smooth_method {smooth_method!r}"

        # Fetch the default value for floor and add-k
        if smooth_value is None:
            smooth_value = BLEU.SMOOTH_DEFAULTS[smooth_method]

        precisions = [0.0 for x in range(max_ngram_order)]

        smooth_mteval = 1.
        effective_order = max_ngram_order
        for n in range(1, max_ngram_order + 1):
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

        # If the system guesses no i-grams, 1 <= i <= max_ngram_order, the BLEU
        # score is 0 (technically undefined). This is a problem for sentence
        # level BLEU or a corpus of short sentences, where systems will get
        # no credit if sentence lengths fall under the max_ngram_order threshold.
        # This fix scales max_ngram_order to the observed maximum order.
        # It is only available through the API and off by default

        if sys_len < ref_len:
            bp = math.exp(1 - ref_len / sys_len) if sys_len > 0 else 0.0
        else:
            bp = 1.0

        score = bp * math.exp(
            sum(map(my_log, precisions[:effective_order])) / effective_order)

        return BLEUScore(
            score, correct, total, precisions, bp, sys_len, ref_len)

    def get_segment_statistics(self, hyp: str, refs: Iterable[str]):
        correct = [0 for n in range(self.max_ngram_order)]
        total = correct[:]

        # Extract n-grams for the hypothesis
        hyp_tokens = hyp.split()
        hyp_len = len(hyp_tokens)
        hyp_ngrams = extract_word_ngrams(hyp_tokens, 1, self.max_ngram_order)

        # Extract n-grams for the reference(s)
        ref_ngrams, ref_len = BLEU.reference_stats(refs, hyp_len)

        # Count the stats
        for ngram in hyp_ngrams.keys():
            order = len(ngram)
            correct[order - 1] += min(hyp_ngrams[ngram], ref_ngrams.get(ngram, 0))
            total[order - 1] += hyp_ngrams[ngram]

        return (correct, total, hyp_len, ref_len)

    def get_statistics(self, hyps: Iterable[str],
                       refs: List[Iterable[str]]) -> List[Tuple[int, ...]]:
        """Reads the corpus and returns sentence-level match statistics for
        quickly re-computing BLEU afterwards, for significance testing.

        :param hyps: An iterable of hypotheses / sentences.
        :param refs: Possibly multiple references per each hypotheses, wrapped
            into a nested Iterable.
        :return: A list of `SampleStat` objects.
        """

        # sanity checks
        if any(len(ref_stream) != len(hyps) for ref_stream in refs):
            raise RuntimeError("System and reference streams have different lengths!")

        if any(line is None for line in hyps):
            raise EOFError("Undefined line in hypotheses stream!")

        stats = []

        tok_count = 0

        for hyp, *cur_refs in zip(hyps, *refs):
            # remove undefined / empty references
            # i.e. we have fewer references for this particular sentence
            lines = [hyp] + [x for x in cur_refs if x is not None and x != ""]

            if len(lines) < 2:
                # we need at least a hypothesis and a non-empty reference
                raise RuntimeError("Found hypothesis with no valid reference sentences.")

            if self.lowercase:
                lines = [x.lower() for x in lines]

            # Check for already-tokenized input problem
            if lines[0].endswith(' .'):
                tok_count += 1

            # Unpack the lines back and tokenize
            hyp, *cur_refs = [self.tokenizer(x.rstrip()) for x in lines]

            # Collect stats
            stats.append(self.get_segment_statistics(hyp, cur_refs))

        if not self.force and tok_count >= 100:
            sacrelogger.warning("That's 100 lines that end in a tokenized period ('.')")
            sacrelogger.warning("It looks like you forgot to detokenize your test data, which may hurt your score.")
            sacrelogger.warning("If you insist your data is detokenized, or don't care, you can suppress this message with '--force'.")

        return stats

    def corpus_score_from_stats(self, stats: List[Tuple[int, ...]],
                                use_effective_order: bool = False) -> BLEUScore:
        # Accumulate the statistics
        all_correct = [0 for n in range(self.max_ngram_order)]
        all_total = all_correct[:]
        all_hyp_len = 0
        all_ref_len = 0

        for (correct, total, hyp_len, ref_len) in stats:
            all_hyp_len += hyp_len
            all_ref_len += ref_len
            for n in range(self.max_ngram_order):
                all_correct[n] += correct[n]
                all_total[n] += total[n]

        # Get BLEUScore object
        return self.compute_bleu(
            all_correct, all_total, all_hyp_len, all_ref_len,
            smooth_method=self.smooth_method, smooth_value=self.smooth_value,
            use_effective_order=use_effective_order)

    def corpus_score(self, hyps: Iterable[str],
                     refs: List[Iterable[str]],
                     use_effective_order: bool = False) -> BLEUScore:
        """Produces BLEU scores along with its sufficient statistics from a source
        against one or more references.

        :param hyps: The system / hypothesis stream (a sequence of segments)
        :param refs: A list of one or more reference streams (each a sequence of segments)
        :param use_effective_order: Account for references that are shorter than the largest n-gram.
        :return: a `BLEUScore` object containing everything you'd want
        """

        stats = self.get_statistics(hyps, refs)
        return self.corpus_score_from_stats(stats)

    def sentence_score(self, hyp: str, refs: Iterable[str],
                       use_effective_order: bool = True) -> BLEUScore:
        """
        Computes BLEU on a single sentence pair.

        Disclaimer: computing BLEU on the sentence level is not its intended use,
        BLEU is a corpus-level metric.

        :param hypothesis: Hypothesis string.
        :param references: List of reference strings.
        :param use_effective_order: Account for references that are shorter than the largest n-gram.
        :return: a `BLEUScore` object containing everything you'd want
        """
        return self.corpus_score([hyp], [[ref] for ref in refs],
                                 use_effective_order=use_effective_order)
