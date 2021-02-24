from typing import List, Sequence

from ..tokenizers.tokenizer_chrf import TokenizerChrf
from ..significance import bootstrap_ci

from .base import Score, Signature
from .helpers import extract_char_ngrams, check_corpus_score_args, check_sentence_score_args

import numpy as np


class CHRFSignature(Signature):
    def __init__(self, args):
        super().__init__(args)

        self._abbr.update({
            'numchars': 'nc',
            'numwords': 'nw',
            'space': 's',
            'case': 'c',
        })

        self.info.update({
            'space': str(self.args['whitespace']).lower(),
            'case': 'lc' if self.args['lowercase'] else 'mixed',
            'numchars': self.args['char_order'],
            'numwords': self.args['word_order'],
        })


class CHRFScore(Score):
    def __init__(self, score: float, char_order: int, word_order: int, beta: int):
        super().__init__(score)

        self.beta = beta
        self._score_string = None
        self.char_order = char_order
        self.word_order = word_order
        self.prefix = f'chrF{self.beta}'

    def format(self, width=2, score_only=False, signature=''):
        if not self._score_string:
            # Being 0-1 scaled, a default width of 1 is too small for chrF
            self._score_string = f'{self.score:.{width + 1}f}'

        if score_only:
            return self._score_string

        pr = f"{self.prefix}+{signature}" if signature else self.prefix
        return f'{pr} = {self._score_string}'


class BootstrapCHRFScore(CHRFScore):
    """A convenience class that computes average chrF score from a collection
    of bootstrap resamples.

    :param scores: A list of `CHRFScore` objects.
    """
    def __init__(self, scores: List[CHRFScore]):
        self.n_bootstrap = len(scores)
        mean_chrf, self.ci = bootstrap_ci([x.score for x in scores])

        super().__init__(
            mean_chrf, scores[0].char_order,
            scores[0].word_order, scores[0].beta)

    def format(self, width=2, score_only=False, signature=''):
        self._score_string = f'{self.score:.{width}f} +/- {self.ci:.5f}'
        return super().format(width, score_only, signature)


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

    def __init__(self, whitespace: bool = False,
                 char_order: int = CHAR_ORDER,
                 word_order: int = WORD_ORDER,
                 beta: float = BETA,
                 lowercase: bool = False,
                 num_refs: int = 1):
        self.name = 'chrf'
        self.beta = beta
        self.char_order = char_order
        self.word_order = word_order
        self.order = self.char_order + self.word_order
        self.num_refs = num_refs
        self.lowercase = lowercase
        self.whitespace = whitespace
        self.signature = CHRFSignature(self.__dict__)
        self.tokenizer = TokenizerChrf(self.lowercase, self.whitespace)

    def compute_chrf(self, statistics: np.ndarray) -> CHRFScore:
        """Computes the chrF++ score from the given match statistics.

        :param statistics: A numpy array with the match statistics.
        :return: a `CHRFScore` object.
        """
        score = 0.0
        avg_recall = 0.0
        avg_precision = 0.0
        effective_order = 0

        statistics = statistics.sum(0).astype(int).tolist()

        for i in range(self.char_order):
            hypotheses_ngrams = statistics[3 * i + 0]
            references_ngrams = statistics[3 * i + 1]
            common_ngrams = statistics[3 * i + 2]
            if hypotheses_ngrams > 0 and references_ngrams > 0:
                avg_precision += common_ngrams / hypotheses_ngrams
                avg_recall += common_ngrams / references_ngrams
                effective_order += 1

        if effective_order == 0:
            avg_precision, avg_recall = 0.0, 0.0
        else:
            avg_precision /= effective_order
            avg_recall /= effective_order

        if avg_precision + avg_recall == 0:
            score = 0.0
        else:
            beta_square = self.beta ** 2
            score = (1 + beta_square) * (avg_precision * avg_recall)
            score /= ((beta_square * avg_precision) + avg_recall)

        return CHRFScore(score, self.char_order, self.word_order, self.beta)

    def _extract_segment_statistics(self, hyp: str, refs: Sequence[str]) -> List[int]:
        """Computes the match statistics given a single hypothesis and multiple references.

        :param hyp: A string representing the hypothesis
        :param refs: An iterable of multiple reference segments
        :return: A flattened list.
        """
        # NOTE: multi-reference not supported yet
        ref = refs[0]

        statistics = [0 for i in range(self.order * 3)]

        # extract character n-grams
        for i in range(self.char_order):
            n = i + 1

            hypothesis_ngrams = extract_char_ngrams(hyp, n)
            reference_ngrams = extract_char_ngrams(ref, n)
            common_ngrams = hypothesis_ngrams & reference_ngrams

            # compute character stats
            statistics[3 * i + 0] = sum(hypothesis_ngrams.values())
            statistics[3 * i + 1] = sum(reference_ngrams.values())
            statistics[3 * i + 2] = sum(common_ngrams.values())

        return statistics

    def _extract_corpus_statistics(self, hyps: Sequence[str],
                                   refs: Sequence[Sequence[str]] = None) -> List[List[int]]:
        """Reads the corpus and returns sentence-level match statistics for
        quickly re-computing BLEU afterwards, for significance testing.

        :param hyps: An iterable of hypotheses / sentences.
        :param refs: Possibly multiple references per each hypotheses, wrapped
            into a nested Sequence.
        :return: A List[List[int]] where each element is a flattened list
            returned by `BLEU.get_segment_statistics()`.
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

            # Unpack the lines back and tokenize
            hyp, *cur_refs = [self.tokenizer(x.rstrip()) for x in lines]

            # Collect stats
            stats.append(self._extract_segment_statistics(hyp, cur_refs))

        return stats

    def corpus_score(self, hyps: Sequence[str],
                     refs: Sequence[Sequence[str]],
                     n_bootstrap: int = 1) -> CHRFScore:
        """
        Computes Chrf on a corpus.

        :param hyps: The system / hypothesis stream (a sequence of segments)
        :param refs: A list of one or more reference streams (each a sequence of segments)
        :param n_bootstrap: If > 1, provides 95% confidence interval around true mean
            using bootstrap resampling with `n_bootstrap` samples with replacement.
        :return: a `CHRFScore` object.
        """
        check_corpus_score_args(hyps, refs)

        # float32 is more efficient than (u)int arrays
        stats = np.array(
            self._extract_corpus_statistics(hyps, refs), dtype='float32')

        if n_bootstrap == 1:
            # Compute the usual BLEU score
            return self.compute_chrf(stats)

        # Get bootstrap estimate & resample
        samples = np.random.choice(
            len(stats), size=(n_bootstrap, len(stats)), replace=True)

        # recompute chrF scores
        scores = [self.compute_chrf(stats[idx]) for idx in samples]

        # Update BLEU signature
        self.signature.update('bootstrap', n_bootstrap)
        return BootstrapCHRFScore(scores)

    def sentence_score(self, hyp: str, refs: Sequence[str]) -> CHRFScore:
        """
        Computes chrF++ on a single sentence pair.

        :param hyp: Hypothesis string.
        :param refs: Reference string(s).
        :return: a `CHRFScore` object.
        """
        check_sentence_score_args(hyp, refs)

        stats = self._extract_segment_statistics(hyp, refs)
        return self.compute_chrf(stats)
