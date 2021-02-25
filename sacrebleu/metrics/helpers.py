from collections import Counter
from typing import List, Sequence


def check_sentence_score_args(hyp: str, refs: Sequence[str]):
    if not isinstance(hyp, str):
        raise RuntimeError('The argument `hyp` should be a string.')

    if not isinstance(refs, Sequence):
        raise RuntimeError('The argument `refs` should be a sequence of strings.')

    if not isinstance(refs[0], str):
        raise RuntimeError('Each element of `refs` should be a string.')


def check_corpus_score_args(hyps: Sequence[str], refs: Sequence[Sequence[str]]):
    if not isinstance(hyps, Sequence):
        raise RuntimeError("`hyps` should be a sequence of strings.")

    if not isinstance(hyps[0], str):
        raise RuntimeError('Each element of `hyps` should be a string.')

    if not isinstance(refs, Sequence):
        raise RuntimeError("`refs` should be a sequence of sequence of strings.")

    if not isinstance(refs[0], Sequence):
        raise RuntimeError("Each element of `refs` should be a sequence of strings.")

    if not isinstance(refs[0][0], str):
        raise RuntimeError("`refs` should be a sequence of sequence of strings.")


def extract_all_word_ngrams(tokens: List[str], min_order: int, max_order: int) -> Counter:
    """Extracts all ngrams (min_order <= n <= max_order) from a sentence.

    :param tokens: A list of tokens.
    :param min_order: Minimum n-gram order.
    :param max_order: Maximum n-gram order.
    :return: a Counter object with n-grams counts.
    """

    ngrams = []

    for n in range(min_order, max_order + 1):
        for i in range(0, len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i: i + n]))

    return Counter(ngrams)


def extract_word_ngrams(tokens: List[str], n: int) -> Counter:
    """Extracts n-grams with order `n` from a list of tokens.

    :param tokens: A list of tokens.
    :param n: The order of n-grams.
    :return: a Counter object with n-grams counts.
    """
    return Counter([' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)])


def extract_char_ngrams(line: str, n: int, include_whitespace: bool = False) -> Counter:
    """Yields counts of character n-grams from a sentence.

    :param line: A segment containing a sequence of words.
    :param n: The order of the n-grams.
    :param include_whitespace: If given, will not strip whitespaces from the line.
    :return: a dictionary containing ngrams and counts
    """
    if not include_whitespace:
        line = ''.join(line.split())

    return Counter([line[i:i + n] for i in range(len(line) - n + 1)])
