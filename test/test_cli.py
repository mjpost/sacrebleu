import json
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "system_output, expected_ter",
    [
        ("one\t\ntwo\ttwo\n", 50.0),
        ("one\tone\ntwo\t\n", 50.0),
        ("one\t\ntwo\t\n", 100.0),
    ],
)
def test_empty_final_system_output(tmp_path, system_output, expected_ter):
    refs = tmp_path / "ref.txt"
    refs.write_text("one\ntwo\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "sacrebleu", str(refs), "-m", "ter", "-f", "json"],
        input=system_output, capture_output=True, text=True, check=True,
    )
    rows = json.loads(result.stdout)
    assert [(row["system"], float(row["TER"])) for row in rows] == [
        ("System 1", 0.0), ("System 2", expected_ter),
    ]


def test_empty_final_reference(tmp_path):
    refs = tmp_path / "refs.tsv"
    refs.write_bytes(b"one\t\r\ntwo\tthree\r\n")
    result = subprocess.run(
        [sys.executable, "-m", "sacrebleu", str(refs), "--num-refs", "2", "-m", "ter", "-f", "json"],
        input="one\nthree\n", capture_output=True, text=True, check=True,
    )
    score = json.loads(result.stdout)
    assert score["score"] == 0.0
    assert score["nrefs"] == "2"
