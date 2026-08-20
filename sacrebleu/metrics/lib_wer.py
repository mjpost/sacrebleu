"""This module implements the word-level edit distance used by the WER metric."""

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


def word_error_rate_stats(words_hyp: list[str], words_ref: list[str]) -> tuple[int, int]:
    """Computes the word-level edit distance (Levenshtein distance, i.e. the
    minimum number of word substitutions, insertions and deletions) between a
    hypothesis and a reference, as used by Word Error Rate (WER).

    Unlike TER, WER does not consider block "shifts" as a single edit.

    :param words_hyp: Tokenized hypothesis.
    :param words_ref: Tokenized reference.
    :return: tuple (number of edits, reference length)
    """
    n_words_ref = len(words_ref)
    n_words_hyp = len(words_hyp)

    if n_words_ref == 0:
        # Special treatment of empty references, consistent with TER.
        return n_words_hyp, 0

    # Rolling two-row dynamic programming; O(n_words_hyp * n_words_ref) time,
    # O(n_words_ref) space.
    previous_row = list(range(n_words_ref + 1))
    for i, word_hyp in enumerate(words_hyp, start=1):
        current_row = [i] + [0] * n_words_ref
        for j, word_ref in enumerate(words_ref, start=1):
            if word_hyp == word_ref:
                current_row[j] = previous_row[j - 1]
            else:
                current_row[j] = 1 + min(
                    previous_row[j - 1],  # substitution
                    previous_row[j],      # deletion
                    current_row[j - 1],   # insertion
                )
        previous_row = current_row

    return previous_row[n_words_ref], n_words_ref
