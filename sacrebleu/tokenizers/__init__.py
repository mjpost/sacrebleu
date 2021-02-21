from .tokenizer_base import BaseTokenizer
from .tokenizer_13a import Tokenizer13a
from .tokenizer_intl import TokenizerV14International
from .tokenizer_zh import TokenizerZh
from .tokenizer_ja_mecab import TokenizerJaMecab
from .tokenizer_char import TokenizerChar


BLEU_TOKENIZERS = {
    'zh': TokenizerZh,
    '13a': Tokenizer13a,
    'none': BaseTokenizer,
    'char': TokenizerChar,
    'intl': TokenizerV14International,
    'ja-mecab': TokenizerJaMecab,
}

DEFAULT_BLEU_TOKENIZER = '13a'
