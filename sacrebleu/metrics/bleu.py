import math
import logging
from collections import Counter
from typing import List, Iterable, Optional

from ..tokenizers import BLEU_TOKENIZERS
from ..utils import my_log
from ..significance import bootstrap_ci
from .base import BaseScore, Signature
from .helpers import extract_word_ngrams

import numpy as np

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

            smooth_str += '[{smooth_val:.2f}]'

        self.info.update({
            'smooth': smooth_str,
            'case': 'lc' if self.args['lowercase'] else 'mixed',
            'tok': self.args['tokenizer_signature'],
            'numrefs': self.args.get('num_refs', '?'),
        })


class BLEUScore(BaseScore):
    """A convenience class to represent BLEU scores (without signature)."""
    def __init__(self, score, precisions, bp, sys_len, ref_len):
        super().__init__(score)

        self.prefix = 'BLEU'
        self.bp = bp
        self.sys_len = int(sys_len)
        self.ref_len = int(ref_len)
        self.precisions = precisions
        self.prec_str = "/".join([f"{p:.1f}" for p in self.precisions])
        self.ratio = self.sys_len / self.ref_len if self.ref_len else 0
        self._score_string = None

    def format(self, width=2, score_only=False, signature=''):
        if not self._score_string:
            self._score_string = f'{self.score:.{width}f}'

        if score_only:
            return self._score_string

        pr = f"{self.prefix}+{signature}" if signature else self.prefix
        s = f'{pr} = {self._score_string} {self.prec_str} '
        s += f'(BP = {self.bp:.3f} ratio = {self.ratio:.3f} '
        s += f'hyp_len = {self.sys_len:d} ref_len = {self.ref_len:d})'
        return s


class BootstrapBLEUScore(BLEUScore):
    """A convenience class that computes average BLEU and other stats from
    a collection of bootstrap resamples.

    :param scores: A list of `BLEUScore` objects.
    """
    def __init__(self, scores: List[BLEUScore]):
        self.n_bootstrap = len(scores)
        bp = sum(map(lambda x: x.bp, scores)) / len(scores)
        sys_len = sum(map(lambda x: x.sys_len, scores)) / len(scores)
        ref_len = sum(map(lambda x: x.ref_len, scores)) / len(scores)
        precisions = np.array([x.precisions for x in scores]).mean(0).tolist()
        mean_bleu, self.ci = bootstrap_ci([x.score for x in scores])

        # Call parent's __init__()
        super().__init__(mean_bleu, precisions, bp, sys_len, ref_len)

    def format(self, width=2, score_only=False, signature=''):
        self._score_string = f'{self.score:.{width}f} +/- {self.ci:.3f}'
        return super().format(width, score_only, signature)


class BLEU:
    """Computes the BLEU metric given hypotheses and references.

    :param lowercase: If True, lowercased BLEU is computed
    :param force: Ignore data that looks already tokenized
    :param tokenize: The tokenizer to use. If None, defaults to language-specific tokenizers with '13a' as the fallback default.
    :param smooth_method: The smoothing method to use ('floor', 'add-k', 'exp' or 'none')
    :param smooth_value: The smoothing value for `floor` and `add-k` methods. `None` falls back to default value.
    :param max_ngram_order: If given, it overrides the maximum n-gram order (default: 4) when computing precisions.
    :param num_refs: The number of references given
    :param trg_lang: An optional language code to raise potential tokenizer warnings.
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

    # mteval-v13a.pl tokenizer unless Chinese or Japanese is provided
    TOKENIZER_DEFAULT = '13a'

    # Some language specific mappings when specific langpair selected
    # through cmdline
    _TOKENIZER_MAP = {
        'zh': 'zh',
        'ja': 'ja-mecab',
    }

    def __init__(self, lowercase: bool = False,
                 force: bool = False,
                 tokenize: Optional[str] = '13a',
                 smooth_method: str = 'exp',
                 smooth_value: Optional[float] = None,
                 max_ngram_order: int = MAX_NGRAM_ORDER,
                 num_refs: int = 1,
                 trg_lang: str = ''):
        self.name = 'bleu'
        self.force = force
        self.num_refs = num_refs
        self.trg_lang = trg_lang
        self.lowercase = lowercase
        self.smooth_value = smooth_value
        self.smooth_method = smooth_method
        self.max_ngram_order = max_ngram_order

        # Sanity check
        assert self.smooth_method in self.SMOOTH_DEFAULTS.keys(), \
            "Unknown smooth_method {self.smooth_method!r}"

        # Default tokenizer assignment
        if tokenize is None:
            # Set `zh` or `ja-mecab` if target language is provided
            tokenize = self._TOKENIZER_MAP.get(self.trg_lang, self.TOKENIZER_DEFAULT)

        if self.trg_lang and self.trg_lang in self._TOKENIZER_MAP:
            best_tokenizer = self._TOKENIZER_MAP[self.trg_lang]
            if self.trg_lang == 'zh' and tokenize != best_tokenizer:
                sacrelogger.warning(
                    "You should use the 'zh' tokenizer for Chinese.")
            if self.trg_lang == 'ja' and tokenize != best_tokenizer:
                sacrelogger.warning(
                    "You should use the 'ja-mecab' tokenizer for Japanese.")

        if tokenize == 'none':
            sacrelogger.warning(
                "You are turning off BLEU's internal tokenizer "
                "presumably to supply your own tokenized files.")
            sacrelogger.warning(
                "Published numbers will not be comparable to other papers.")

        # Create the tokenizer
        self.tokenizer = BLEU_TOKENIZERS[tokenize]()

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
        ngrams = Counter()

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

            # Merge counts: Union will keep the max of two
            ngrams |= extract_word_ngrams(ref_tokens, 1, max_ngram_order)

        return ngrams, closest_ref_len

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
            score, precisions, bp, sys_len, ref_len)

    def _get_ngram_counts(self, ngrams: Counter) -> List[int]:
        """Returns a list of n-gram counts.

        :param ngrams: A Counter with n-gram tuples and their counts as keys & values.
        :return: A list of `max_ngram_order` integers that represent the counts.
        """
        c = [0 for i in range(self.max_ngram_order)]
        for key, count in ngrams.items():
            c[len(key) - 1] += count
        return c

    def get_segment_statistics(self, hyp: str, refs: Iterable[str]) -> List[int]:
        """Computes the match statistics given a single hypothesis and multiple references.

        :param hyp: A string representing the hypothesis
        :param refs: An iterable of multiple reference segments
        :return: A flattened list where the first two integers denote the
            hypothesis and reference lengths. The next `max_gram_order` elements
            give the number of correct n-gram matches and the final `max_ngram_order`
            elements give the number of total n-grams in the hypothesis.
        """

        # Extract n-grams for the hypothesis
        hyp_tokens = hyp.split()
        hyp_len = len(hyp_tokens)
        hyp_ngrams = extract_word_ngrams(hyp_tokens, 1, self.max_ngram_order)

        # Extract n-grams for the reference(s)
        ref_ngrams, ref_len = BLEU.reference_stats(refs, hyp_len)

        # Count the stats
        matched_ngrams = hyp_ngrams & ref_ngrams
        correct = self._get_ngram_counts(matched_ngrams)
        total = self._get_ngram_counts(hyp_ngrams)

        # Return a flattened list for efficient computation
        return [hyp_len, ref_len] + correct + total

    def get_corpus_statistics(self, hyps: Iterable[str],
                              refs: List[Iterable[str]] = None) -> List[List[int]]:
        """Reads the corpus and returns sentence-level match statistics for
        quickly re-computing BLEU afterwards, for significance testing.

        :param hyps: An iterable of hypotheses / sentences.
        :param refs: Possibly multiple references per each hypotheses, wrapped
            into a nested Iterable.
        :return: A List[List[int]] where each element is a flattened list
            returned by `BLEU.get_segment_statistics()`.
        """

        # sanity checks
        if any(len(ref_stream) != len(hyps) for ref_stream in refs):
            raise RuntimeError("System and reference streams have different lengths!")

        if any(line is None for line in hyps):
            raise EOFError("Undefined line in hypotheses stream!")

        tok_count = 0
        stats = []

        cur_refs: List[str]

        for idx, (hyp, *cur_refs) in enumerate(zip(hyps, *refs)):
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

    def corpus_score_from_stats(self, stats: np.ndarray,
                                use_effective_order: bool = False) -> BLEUScore:
        """Computes the final BLEU score given the pre-computed corpus statistics.

        :param stats: A list segment-level statistics
        :return: BLEUScore object.
        """
        # Accumulate the statistics
        sum_stats = stats.sum(0).astype(np.uint).tolist()

        # Get BLEUScore object
        return self.compute_bleu(
            correct=sum_stats[2: 2 + self.max_ngram_order],
            total=sum_stats[2 + self.max_ngram_order:],
            sys_len=sum_stats[0], ref_len=sum_stats[1],
            smooth_method=self.smooth_method, smooth_value=self.smooth_value,
            use_effective_order=use_effective_order)

    def corpus_score(self, hyps: Iterable[str],
                     refs: List[Iterable[str]],
                     use_effective_order: bool = False,
                     n_bootstrap: int = 1) -> BLEUScore:
        """Produces BLEU scores along with its sufficient statistics from a source
        against one or more references.

        :param hyps: The system / hypothesis stream (a sequence of segments)
        :param refs: A list of one or more reference streams (each a sequence of segments)
        :param use_effective_order: Account for references that are shorter than the largest n-gram.
        :return: a `BLEUScore` object containing everything you'd want
        """

        # float32 is more efficient than (u)int arrays
        stats = np.array(
            self.get_corpus_statistics(hyps, refs), dtype='float32')

        # Get bootstrap estimate
        if n_bootstrap > 1:
            # Update signature
            self.signature.update('bootstrap', n_bootstrap)
            # Resample with replacement
            samples = np.random.choice(
                len(stats), size=(n_bootstrap, len(stats)), replace=True)
            scores = [self.corpus_score_from_stats(stats[idx]) for idx in samples]
            return BootstrapBLEUScore(scores)
        else:
            # Usual BLEU score
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
