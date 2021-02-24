from itertools import zip_longest
from typing import List, Union, Sequence

from ..tokenizers.tokenizer_chrf import TokenizerChrf

from .base import Score, Signature
from .helpers import extract_char_ngrams


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
    def __init__(self, score, beta, order):
        super().__init__(score)

        self.beta = beta
        self.char_order = order
        self.prefix = f'chrF{self.beta}'

    def format(self, width=2, score_only=False, signature=''):
        # Being 0-1 scaled, a default width of 1 is too small for chrF
        if score_only:
            return f'{self.score:.{width + 1}f}'

        prefix = f"{self.prefix}+{signature}" if signature else self.prefix
        return f'{prefix} = {self.score:.{width + 1}f}'


class CHRF:
    """Computes the chrF++ metric given hypotheses and references.

    :param whitespace: If True, includes the whitespace character in chrF computation.
    :param char_order: chrF character order
    :param word_order: chrF word order
    :param beta: chrF Beta parameter
    :param lowercase: Lowercase sentences prior computation
    :param num_refs: Number of references (not functional for chrF as of now)
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
        self.num_refs = num_refs
        self.lowercase = lowercase
        self.whitespace = whitespace
        self.signature = CHRFSignature(self.__dict__)
        self.tokenizer = TokenizerChrf(self.lowercase, self.whitespace)

    @staticmethod
    def compute_chrf(statistics: List[int],
                     char_order: int,
                     beta: float) -> CHRFScore:
        score = 0.0
        avg_recall = 0.0
        avg_precision = 0.0
        effective_order = 0

        for i in range(char_order):
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
            beta_square = beta ** 2
            score = (1 + beta_square) * (avg_precision * avg_recall)
            score /= ((beta_square * avg_precision) + avg_recall)

        return CHRFScore(score, beta, char_order)

    def get_sentence_statistics(self, hypothesis: str,
                                references: List[str]) -> List[int]:
        # NOTE: multi-reference not supported yet
        reference = references[0]

        hypothesis = self.tokenizer(hypothesis)
        reference = self.tokenizer(reference)
        statistics = [0] * (self.char_order * 3)
        for i in range(self.char_order):
            n = i + 1
            hypothesis_ngrams = extract_char_ngrams(hypothesis, n)
            reference_ngrams = extract_char_ngrams(reference, n)
            common_ngrams = hypothesis_ngrams & reference_ngrams
            statistics[3 * i + 0] = sum(hypothesis_ngrams.values())
            statistics[3 * i + 1] = sum(reference_ngrams.values())
            statistics[3 * i + 2] = sum(common_ngrams.values())
        return statistics

    def sentence_score(self, hyp: str, refs: Sequence[str]) -> CHRFScore:
        """
        Computes ChrF on a single sentence pair.

        :param hyp: Hypothesis string.
        :param refs: Reference string(s).
        :return: Chrf score.
        """
        assert not isinstance(references, str), \
            "sentence_score needs a list of references, not a single string"
        stats = self.get_sentence_statistics(hypothesis, references)
        return self.compute_chrf(stats, self.char_order, self.beta)

    def corpus_score(self, hyps: Sequence[str],
                     refs: Sequence[Sequence[str]]) -> CHRFScore:
        """
        Computes Chrf on a corpus.

        :param hypotheses: Stream of hypotheses.
        :param references: Stream of references.
        :return: Chrf score.
        """

        # Add some robustness to the input arguments
        if isinstance(sys_stream, str):
            sys_stream = [sys_stream]

        if isinstance(ref_streams, str):
            ref_streams = [[ref_streams]]

        corpus_statistics = [0] * (self.char_order * 3)

        fhs = [sys_stream] + ref_streams
        for lines in zip_longest(*fhs):
            if None in lines:
                raise EOFError("Source and reference streams have different lengths!")

            # Unpack
            hypothesis, *refs = lines

            statistics = self.get_sentence_statistics(hypothesis, refs)
            for i in range(len(statistics)):
                corpus_statistics[i] += statistics[i]

        return self.compute_chrf(corpus_statistics, self.char_order, self.beta)
