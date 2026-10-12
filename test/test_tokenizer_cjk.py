import json
import subprocess
import sys

import pytest

from sacrebleu import corpus_bleu
from sacrebleu.metrics import BLEU
from sacrebleu.tokenizers.tokenizer_cjk import TokenizerCJK
from sacrebleu.tokenizers.tokenizer_zh import TokenizerZh


@pytest.mark.parametrize('text, expected', [
    ('ひらがなカタカナ John Smith 中文', 'ひ ら が な カ タ カ ナ John Smith 中 文'),
    ('JohnひらSmithカナ', 'John ひ ら Smith カ ナ'),
    ('カタカナ、ひらがな。', 'カ タ カ ナ 、 ひ ら が な 。'),
    ('かな, 3.14 test-name!', 'か な , 3.14 test-name !'),
    ('\u3040\u309f\u30a0\u30ff', '\u3040 \u309f \u30a0 \u30ff'),
    ('か\u3099ガ', 'か \u3099 ガ'),  # No Unicode normalization.
])
def test_cjk_tokenization(text, expected):
    assert TokenizerCJK()(text) == expected


@pytest.mark.parametrize('text', [
    '', ' \t\n ', '  John\tSmith\n', '中文，测试。', 'ｶﾀｶﾅ',
    'Price: 3.14, 1,000 test-name!', '한국어', '\U00020000\U00020001',
    '\u303f\u3100', '\u31f0\u31ff',
])
def test_cjk_preserves_zh_behavior_outside_kana(text):
    assert TokenizerCJK()(text) == TokenizerZh()(text)


def test_cjk_adds_only_requested_range():
    for codepoint in range(0x10000):
        char = chr(codepoint)
        assert TokenizerCJK._is_chinese_char(char) == (
            0x3040 <= codepoint <= 0x30ff or TokenizerZh._is_chinese_char(char)
        )


def test_cjk_does_not_change_zh():
    text = 'ひらがなカタカナ John Smith 中文'
    assert TokenizerZh()(text) == 'ひらがなカタカナ John Smith 中 文'
    assert TokenizerCJK()(text) == 'ひ ら が な カ タ カ ナ John Smith 中 文'
    assert TokenizerZh()(text) == 'ひらがなカタカナ John Smith 中 文'
    assert TokenizerZh().signature() == 'zh'


def test_cjk_bleu_api_and_signature():
    hypothesis = ['ひらがなカタカナ John Smith 中文']
    references = [['ひ ら が な カ タ カ ナ John Smith 中 文']]
    bleu = BLEU(tokenize='cjk')
    assert bleu.corpus_score(hypothesis, references).score == pytest.approx(100)
    assert 'tok:cjk' in str(bleu.get_signature())
    assert corpus_bleu(hypothesis, references, tokenize='cjk').score == pytest.approx(100)
    assert BLEU().tokenizer.signature() == '13a'
    assert BLEU(trg_lang='zh').tokenizer.signature() == 'zh'


def test_cjk_cli(tmp_path, monkeypatch):
    monkeypatch.delenv('SACREBLEU_FORMAT', raising=False)
    reference = tmp_path / 'reference.txt'
    reference.write_text('ひ ら が な カ タ カ ナ John Smith 中 文\n', encoding='utf-8')
    result = subprocess.run(
        [sys.executable, '-m', 'sacrebleu', str(reference), '--tokenize', 'cjk', '--format', 'json'],
        input='ひらがなカタカナ John Smith 中文\n',
        encoding='utf-8', capture_output=True, check=True,
    )
    score = json.loads(result.stdout)
    assert score['score'] == pytest.approx(100)
    assert 'tok:cjk' in score['signature']
