"""The implementation of the Word Error Rate (WER) metric."""

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..utils import sum_of_lists
from .base import Metric, Score, Signature
from .lib_wer import word_error_rate_stats


class WERSignature(Signature):
    """A convenience class to represent the reproducibility signature for WER.

    :param args: key-value dictionary passed from the actual metric instance.
    """
    def __init__(self, args: dict):
        """`WERSignature` initializer."""
        super().__init__(args)
        self._abbr.update({
            'case': 'c',
        })

        self.info.update({
            'case': 'mixed' if args['case_sensitive'] else 'lc',
        })


class WERScore(Score):
    """A convenience class to represent WER scores.

    :param score: The WER score.
    :param num_edits: The cumulative number of edits.
    :param ref_length: The cumulative average reference length.
    """
    def __init__(self, score: float, num_edits: float, ref_length: float):
        """`WERScore` initializer."""
        super().__init__('WER', score)
        self.num_edits = int(num_edits)
        self.ref_length = ref_length


class WER(Metric):
    """Word error rate (WER). The word-level edit distance between a
    hypothesis and the closest-matching reference, normalized by the
    reference length, expressed as a percentage.

    Unlike TER, WER does not treat contiguous word-block reorderings
    ("shifts") as a single edit -- it is plain word-level Levenshtein
    distance, which keeps the implementation simple and dependency-free
    at the cost of being slower than a C-backed edit-distance library for
    very long sequences.

    :param case_sensitive: If `True`, does not lowercase sentences.
    :param references: A sequence of reference documents with document being
        defined as a sequence of reference strings. If given, the reference info
        will be pre-computed and cached for faster re-computation across many systems.
    """

    _SIGNATURE_TYPE = WERSignature

    def __init__(self, case_sensitive: bool = False,
                 references: Sequence[Sequence[str]] | None = None):
        """`WER` initializer."""
        super().__init__()

        self.case_sensitive = case_sensitive

        if references is not None:
            self._ref_cache = self._cache_references(references)

    def _preprocess_segment(self, sent: str) -> str:
        """Given a sentence, apply case-folding if enabled.

        :param sent: The input sentence string.
        :return: The pre-processed output string.
        """
        sent = sent.rstrip()
        return sent if self.case_sensitive else sent.lower()

    def _compute_score_from_stats(self, stats: list[float]) -> WERScore:
        """Computes the final score from already aggregated statistics.

        :param stats: A list or numpy array of segment-level statistics.
        :return: A `WERScore` object.
        """
        total_edits, sum_ref_lengths = stats[0], stats[1]

        if sum_ref_lengths > 0:
            score = total_edits / sum_ref_lengths
        elif total_edits > 0:
            score = 1.0  # empty reference(s) and non-empty hypothesis
        else:
            score = 0.0  # both reference(s) and hypothesis are empty

        return WERScore(100 * score, total_edits, sum_ref_lengths)

    def _aggregate_and_compute(self, stats: list[list[float]]) -> WERScore:
        """Computes the final WER score given the pre-computed corpus statistics.

        :param stats: A list of segment-level statistics
        :return: A `WERScore` instance.
        """
        return self._compute_score_from_stats(sum_of_lists(stats))

    def _compute_segment_statistics(
            self, hypothesis: str, ref_kwargs: dict) -> list[float]:
        """Given a (pre-processed) hypothesis sentence and already computed
        reference words, returns the segment statistics required to compute
        the full WER score.

        :param hypothesis: Hypothesis sentence.
        :param ref_kwargs: A dictionary with `ref_words` key which is a list
        where each sublist contains reference words.
        :return: A two-element list that contains the 'minimum number of edits'
        and 'the average reference length'.
        """
        ref_lengths = 0
        best_num_edits = int(1e16)

        words_hyp = hypothesis.split()

        # Iterate the references
        ref_words = ref_kwargs['ref_words']
        for words_ref in ref_words:
            num_edits, ref_len = word_error_rate_stats(words_hyp, words_ref)
            ref_lengths += ref_len
            best_num_edits = min(best_num_edits, num_edits)

        avg_ref_len = ref_lengths / len(ref_words)
        return [best_num_edits, avg_ref_len]

    def _extract_reference_info(self, refs: Sequence[str]) -> dict[str, Any]:
        """Given a list of reference segments, applies pre-processing and
        returns list of tokens for each reference.

        :param refs: A sequence of strings.
        :return: A dictionary that will be passed to `_compute_segment_statistics()`
        through keyword arguments.
        """
        ref_words = []

        for ref in refs:
            ref_words.append(self._preprocess_segment(ref).split())

        return {'ref_words': ref_words}
