import string

from typing import List, Sequence
from collections import Counter

from ..significance import bootstrap_ci

from .base import Score, Signature
from .helpers import check_corpus_score_args, check_sentence_score_args
from .helpers import extract_char_ngrams, extract_word_ngrams

import numpy as np


class CHRFSignature(Signature):
    def __init__(self, args):
        super().__init__(args)

        self._abbr.update({
            'nchars': 'nc',
            'nwords': 'nw',
            'space': 's',
            'case': 'c',
        })

        self.info.update({
            'space': str(self.args['whitespace']).lower(),
            'case': 'lc' if self.args['lowercase'] else 'mixed',
            'nchars': self.args['char_order'],
            'nwords': self.args['word_order'],
        })


class CHRFScore(Score):
    def __init__(self, score: float, char_order: int,
                 word_order: int, beta: int,
                 n_bootstrap: int = 1, ci: float = 0.0):
        super().__init__(score)

        self.ci = ci
        self.beta = beta
        self._score_string = None
        self.char_order = char_order
        self.word_order = word_order
        self.n_bootstrap = n_bootstrap
        self.prefix = f'chrF{self.beta}'

    @staticmethod
    def average_score(scores):
        """Compute averages across bootstrap resample scores / stats

        :param scores: A list of `CHRFScore` objects for each bootstrap resample.
        :return: a `CHRFScore` object reflecting the estimate scores.
        """
        mean_chrf, ci = bootstrap_ci([x.score for x in scores])
        return CHRFScore(mean_chrf, char_order=scores[0].char_order,
                         word_order=scores[0].word_order,
                         beta=scores[0].beta, n_bootstrap=len(scores),
                         ci=ci)

    def format(self, width=2, score_only=False, signature=''):
        sc = f'{self.score:.{width}f}'
        if self.n_bootstrap > 1:
            sc += f' (+/- {self.ci:.3f})'

        if score_only:
            return sc

        pr = f"{self.prefix}+{signature}" if signature else self.prefix
        return f'{pr} = {sc}'


class CHRF:
    """Computes the chrF(++) metric given hypotheses and references.

    :param whitespace: If True, includes the whitespace character in chrF computation.
    :param char_order: chrF character order.
    :param word_order: chrF++ word order.
    :param beta: chrF Beta parameter.
    :param lowercase: Lowercase sentences prior computation.
    :param num_refs: Number of references.
    """

    # Maximum character n-gram order to take into account
    CHAR_ORDER = 6

    # chrF++ additionally takes into account some of the word n-grams
    WORD_ORDER = 0

    # Defaults to 2 (per http://www.aclweb.org/anthology/W16-2341)
    BETA = 2

    # Cache punctuations for chrF++' punctuation stripper
    _PUNCTS = set(string.punctuation)

    def __init__(self, whitespace: bool = False,
                 char_order: int = CHAR_ORDER,
                 word_order: int = WORD_ORDER,
                 beta: float = BETA,
                 lowercase: bool = False,
                 num_refs: int = 1):
        self.name = 'chrf++' if word_order > 0 else 'chrf'
        self.beta = beta
        self.char_order = char_order
        self.word_order = word_order
        self.order = self.char_order + self.word_order
        self.num_refs = num_refs
        self.lowercase = lowercase
        self.whitespace = whitespace
        self.signature = CHRFSignature(self.__dict__)

    @staticmethod
    def _get_match_statistics(hyp_ngrams: Counter, ref_ngrams: Counter) -> List[int]:
        """Computes the match statistics between hypothesis and reference n-grams.

        :param hyp_ngrams: A `Counter` holding hypothesis n-grams.
        :param ref_ngrams: A `Counter` holding reference n-grams.
        :return: A list of three numbers denoting hypothesis n-gram count,
            reference n-gram count and the intersection count.
        """
        match_ngrams = hyp_ngrams & ref_ngrams

        return [
            # Don't count hits if no reference exists for that n-gram
            sum(hyp_ngrams.values()) if ref_ngrams else 0,
            sum(ref_ngrams.values()),
            sum(match_ngrams.values()),
        ]

    def _remove_punctuation(self, sent: str) -> List[str]:
        """Separates out punctuations from beginning and end of words for chrF++.
        Adapted from https://github.com/m-popovic/chrF

        :param sent: A string representing a sentence.
        :return: A list of words.
        """
        tokenized = []
        for w in sent.split():
            if len(w) == 1:
                tokenized.append(w)
            else:
                # NOTE: This splits '(hi)' to '(hi' and ')' (issue #124)
                if w[-1] in self._PUNCTS:
                    tokenized += [w[:-1], w[-1]]
                elif w[0] in self._PUNCTS:
                    tokenized += [w[0], w[1:]]
                else:
                    tokenized.append(w)
        return tokenized

    def _get_f_score(self, statistics: List[int]) -> float:
        """Compute the chrF score given the n-gram match statistics.

        :param statistics: A flattened list of 3 * (`char_order` + `word_order`)
            elements giving the [hyp, ref, match] counts for each order.
        :return: The final f_beta score.
        """
        eps = 1e-16
        score = 0.0
        factor = self.beta ** 2

        _par_iter = zip(statistics[0::3], statistics[1::3], statistics[2::3])
        for (n_hyp, n_ref, n_match) in _par_iter:
            prec = n_match / n_hyp if n_hyp > 0 else eps
            rec = n_match / n_ref if n_ref > 0 else eps
            denom = factor * prec + rec
            if denom > 0:
                score += (1 + factor) * prec * rec / denom
            else:
                score += eps

        return 100 * score / self.order

    def compute_chrf(self, statistics: List[int]) -> CHRFScore:
        """Computes the chrF++ score from already aggregated match statistics.

        :param statistics: A list of integers for character and word n-gram
            matches. Each triplet in the list denote hypotheses ngrams,
            reference ngrams and matched ngrams counts, respectively.
        :return: a `CHRFScore` object.
        """
        score = self._get_f_score(statistics)
        return CHRFScore(score, self.char_order, self.word_order, self.beta)

    def _extract_segment_statistics(self, hyp: str, refs: Sequence[str]) -> List[int]:
        """Computes the match statistics given a single hypothesis and multiple references.

        :param hyp: A string representing the hypothesis
        :param refs: An iterable of multiple reference segments
        :return: A flattened list.
        """
        statistics = []

        # extract character n-grams
        for n in range(self.char_order):
            hyp_ngrams = extract_char_ngrams(hyp, n + 1, self.whitespace)
            # NOTE: multi-ref
            ref_ngrams = extract_char_ngrams(refs[0], n + 1, self.whitespace)
            statistics.extend(self._get_match_statistics(hyp_ngrams, ref_ngrams))

        # Check chrF++ mode
        if self.word_order > 0:
            # Primitive tokenization: separate out punctuations
            hyp = self._remove_punctuation(hyp)
            refs = [self._remove_punctuation(ref) for ref in refs]

        # chrF++ takes into account word n-grams too
        for n in range(self.word_order):
            hyp_ngrams = extract_word_ngrams(hyp, n + 1)
            # NOTE: Multi-ref
            ref_ngrams = extract_word_ngrams(refs[0], n + 1)
            statistics.extend(self._get_match_statistics(hyp_ngrams, ref_ngrams))

        return statistics

    def _extract_corpus_statistics(self, hyps: Sequence[str],
                                   refs: Sequence[Sequence[str]] = None) -> np.ndarray:
        """Reads the corpus and returns sentence-level match statistics for
        quickly re-computing BLEU afterwards, for significance testing.

        :param hyps: An iterable of hypotheses / sentences.
        :param refs: Possibly multiple references per each hypotheses, wrapped
            into a nested Sequence.
        :return: A numpy matrix where each row is a statistics vector as
            returned by `CHRF._extract_segment_statistics()`.
        """

        # sanity checks
        if any(len(ref_stream) != len(hyps) for ref_stream in refs):
            raise RuntimeError("System and reference streams have different lengths!")

        if any(line is None for line in hyps):
            raise EOFError("Undefined line in hypotheses stream!")

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

            # Unpack the lines back
            hyp, *cur_refs = lines

            # Collect stats
            stats.append(self._extract_segment_statistics(hyp, cur_refs))

        return np.array(stats, dtype='float32')

    def sentence_score(self, hyp: str, refs: Sequence[str]) -> CHRFScore:
        """
        Computes chrF++ on a single sentence pair.

        :param hyp: Hypothesis string.
        :param refs: Reference string(s).
        :return: a `CHRFScore` object.
        """
        check_sentence_score_args(hyp, refs)

        return self.compute_chrf(self._extract_segment_statistics(hyp, refs))

    def corpus_score(self, hyps: Sequence[str],
                     refs: Sequence[Sequence[str]],
                     n_bootstrap: int = 1) -> CHRFScore:
        """
        Computes chrF++ on a corpus.

        :param hyps: The system / hypothesis stream (a sequence of segments)
        :param refs: A list of one or more reference streams (each a sequence of segments)
        :param n_bootstrap: If > 1, provides 95% confidence interval around true mean
            using bootstrap resampling with `n_bootstrap` samples with replacement.
        :return: a `CHRFScore` object.
        """
        check_corpus_score_args(hyps, refs)

        # Get the statistics
        stats = self._extract_corpus_statistics(hyps, refs)

        if n_bootstrap == 1:
            # Aggregate stats
            stats = stats.sum(0).astype(int).tolist()

            # Compute the usual CHRF score
            return self.compute_chrf(stats)

        # Get bootstrap estimate & resample
        samples = np.random.choice(
            len(stats), size=(n_bootstrap, len(stats)), replace=True)

        # recompute chrF scores
        scores = []
        for idx_list in samples:
            agg_stats = stats[idx_list].sum(0).astype(int).tolist()
            scores.append(self.compute_chrf(agg_stats))

        # Update signature
        self.signature.update('bstrap', n_bootstrap)
        return CHRFScore.average_score(scores)


