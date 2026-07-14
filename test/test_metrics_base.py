from contextlib import nullcontext as does_not_raise

import pytest

from sacrebleu.metrics.base import Metric


class ConcreteTestMetric(Metric):
    """Concrete test metric so the ABC can be instantiated."""

    def _aggregate_and_compute(self, *args, **kwargs): ...

    def _compute_score_from_stats(self, *args, **kwargs): ...

    def _preprocess_segment(self, *args, **kwargs): ...

    def _extract_reference_info(self, *args, **kwargs): ...

    def _compute_segment_statistics(self, *args, **kwargs): ...


@pytest.mark.parametrize(
    "hyp, refs, expected_context",
    [
        # valid
        pytest.param(
            "h1",
            ["r1"],
            does_not_raise(),
            id="valid-minimal",
        ),
        pytest.param(
            "h1",
            ["r1", "r2"],
            does_not_raise(),
            id="valid-multi-reference",
        ),
        pytest.param(
            "h1",
            ("r1",),
            does_not_raise(),
            id="valid-tuple",
        ),
        pytest.param(
            "",
            ["r1"],
            does_not_raise(),
            id="valid-empty-hyp-string",
        ),
        pytest.param(
            "h1",
            ["r1", None],
            does_not_raise(),
            id="valid-none-reference-element",
        ),
        pytest.param(
            "h1",
            [None],
            does_not_raise(),
            id="valid-none-only-reference",
        ),
        # invalid hyp
        pytest.param(
            123,
            ["r1"],
            pytest.raises(TypeError, match="The argument `hyp` should be a string"),
            id="hyp-int",
        ),
        pytest.param(
            None,
            ["r1"],
            pytest.raises(TypeError, match="The argument `hyp` should be a string"),
            id="hyp-none",
        ),
        pytest.param(
            ["h1"],
            ["r1"],
            pytest.raises(TypeError, match="The argument `hyp` should be a string"),
            id="hyp-is-sequence",
        ),
        # invalid refs
        pytest.param(
            "h1",
            "r1",
            pytest.raises(
                TypeError, match="The argument `refs` should be a sequence of strings"
            ),
            id="refs-bare-string",
        ),
        pytest.param(
            "h1",
            b"r1",
            pytest.raises(
                TypeError, match="The argument `refs` should be a sequence of strings"
            ),
            id="refs-bytes",
        ),
        pytest.param(
            "h1",
            123,
            pytest.raises(
                TypeError, match="The argument `refs` should be a sequence of strings"
            ),
            id="refs-not-sequence",
        ),
        pytest.param(
            "h1",
            [],
            pytest.raises(TypeError, match="The argument `refs` should not be empty"),
            id="refs-empty",
        ),
        pytest.param(
            "h1",
            [123],
            pytest.raises(TypeError, match="Each element of `refs` should be a string"),
            id="refs-element-not-string",
        ),
        pytest.param(
            "h1",
            ["r1", 5],
            pytest.raises(TypeError, match="Each element of `refs` should be a string"),
            id="refs-bad-later-element",
        ),
        # both invalid
        pytest.param(
            123,
            "r1",
            pytest.raises(TypeError, match="The argument `hyp` should be a string"),
            id="both-invalid-hyp-message-wins",
        ),
    ],
)
def test_check_sentence_score_args(hyp, refs, expected_context):
    metric = ConcreteTestMetric()
    with expected_context:
        metric._check_sentence_score_args(hyp=hyp, refs=refs)


@pytest.mark.parametrize(
    "hyps, refs, expected_context",
    [
        # valid hyps and refs
        pytest.param(
            ["h1"],
            [["r1a"]],
            does_not_raise(),
            id="valid-minimal",
        ),
        pytest.param(
            ("h1",),
            (("r1a",),),
            does_not_raise(),
            id="valid-tuples",
        ),
        pytest.param(
            ["h1"],
            [["r1a"], ["r2"]],
            does_not_raise(),
            id="valid-multi-reference-single-segment",
        ),
        pytest.param(
            ["h1", "h2"],
            [["r1a", "r2a"], ["r1b", "r2b"]],
            does_not_raise(),
            id="valid-multi-stream-equal-length",
        ),
        pytest.param(
            ["h1", "h2"],
            [["r1a", "r2a"], ["r1b", None]],
            does_not_raise(),
            id="valid-none-padding-equal-length",
        ),
        pytest.param(["h1"], None, does_not_raise(), id="valid-refs-none"),
        pytest.param(
            ["h1"], [[None]], does_not_raise(), id="valid-none-reference-element"
        ),
        # invalid hyps
        pytest.param(
            "h1",
            None,
            pytest.raises(TypeError, match="`hyps` should be a sequence of strings"),
            id="hyps-bare-string",
        ),
        pytest.param(
            b"h1",
            None,
            pytest.raises(TypeError, match="`hyps` should be a sequence of strings"),
            id="hyps-bare-bytes",
        ),
        pytest.param(
            123,
            None,
            pytest.raises(TypeError, match="`hyps` should be a sequence of strings"),
            id="hyps-int",
        ),
        pytest.param(
            None,
            None,
            pytest.raises(TypeError, match="`hyps` should be a sequence of strings"),
            id="hyps-none",
        ),
        pytest.param(
            [],
            None,
            pytest.raises(TypeError, match="`hyps` should not be empty"),
            id="hyps-empty",
        ),
        pytest.param(
            [123],
            None,
            pytest.raises(TypeError, match="Each element of `hyps` should be a string"),
            id="hyps-non-string-element",
        ),
        pytest.param(
            ["hyp1", 123],
            None,
            pytest.raises(TypeError, match="Each element of `hyps` should be a string"),
            id="hyps-non-string-later-element",
        ),
        pytest.param(
            [None],
            None,
            pytest.raises(TypeError, match="Undefined line in hypotheses stream"),
            id="hyps-first-element-none",
        ),
        pytest.param(
            ["h1", None],
            None,
            pytest.raises(TypeError, match="Undefined line in hypotheses stream"),
            id="hyps-trailing-none",
        ),
        # invalid refs - types
        pytest.param(
            ["h1"],
            "r1a",
            pytest.raises(
                TypeError,
                match="Each element of `refs` should be a sequence of strings",
            ),
            id="refs-bare-string",
        ),
        pytest.param(
            ["h1"],
            b"r1",
            pytest.raises(
                TypeError,
                match="Each element of `refs` should be a sequence of strings",
            ),
            id="refs-bare-bytes",
        ),
        pytest.param(
            ["h1"],
            123,
            pytest.raises(
                TypeError, match="`refs` should be a sequence of sequence of strings"
            ),
            id="refs-int",
        ),
        pytest.param(
            ["h1"],
            [],
            pytest.raises(TypeError, match="`refs` should not be empty"),
            id="refs-empty",
        ),
        pytest.param(
            ["h1"],
            ["r1a"],
            pytest.raises(
                TypeError,
                match="Each element of `refs` should be a sequence of strings",
            ),
            id="refs-flat-list-of-strings",
        ),
        pytest.param(
            ["h1"],
            [123],
            pytest.raises(
                TypeError,
                match="Each element of `refs` should be a sequence of strings",
            ),
            id="refs-element-not-sequence",
        ),
        pytest.param(
            ["h1"],
            [[123]],
            pytest.raises(
                TypeError, match="`refs` should be a sequence of sequence of strings"
            ),
            id="refs-inner-element-not-string",
        ),
        # invalid refs - lengths
        pytest.param(
            ["h1", "h2", "h3"],
            [["r1a", "r2a"]],
            pytest.raises(TypeError, match="same length as `hyps`"),
            id="refs-stream-too-short",
        ),
        pytest.param(
            ["h1", "h2"],
            [["r1a", "r2a", "r3a"]],
            pytest.raises(TypeError, match="same length as `hyps`"),
            id="refs-stream-too-long",
        ),
        pytest.param(
            ["h1", "h2"],
            [["r1a", "r2a"], ["r1b"]],
            pytest.raises(TypeError, match="same length as `hyps`"),
            id="refs-streams-unequal",
        ),
        pytest.param(
            ["h1"],
            [[]],
            pytest.raises(TypeError, match="same length as `hyps`"),
            id="refs-empty-inner-stream",
        ),
        # per-stream validation
        pytest.param(
            ["h1", "h2"],
            [["r1a", "r2a"], "bad"],
            pytest.raises(
                TypeError,
                match="Each element of `refs` should be a sequence of strings",
            ),
            id="refs-later-stream-bare-string",
        ),
        pytest.param(
            ["h1"],
            [["r1a"], 5],
            pytest.raises(
                TypeError,
                match="Each element of `refs` should be a sequence of strings",
            ),
            id="refs-later-sequence-not-sequence",
        ),
        pytest.param(
            ["h1", "h2"],
            [["r1a", "r2a"], ["r1b", 99]],
            pytest.raises(
                TypeError, match="`refs` should be a sequence of sequence of strings"
            ),
            id="refs-later-stream-non-string-element",
        ),
        # both invalid
        pytest.param(
            "h1",
            "r1a",
            pytest.raises(TypeError, match="`hyps` should be a sequence of strings"),
            id="both-invalid-hyps-message-wins",
        ),
    ],
)
def test_check_corpus_score_args(hyps, refs, expected_context):
    metric = ConcreteTestMetric()
    with expected_context:
        metric._check_corpus_score_args(hyps=hyps, refs=refs)
