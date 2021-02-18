from .tokenizer_base import BaseTokenizer
from .tokenizer_13a import Tokenizer13a
from .tokenizer_intl import TokenizerV14International
from .tokenizer_zh import TokenizerZh
from .tokenizer_ja_mecab import TokenizerJaMecab
from .tokenizer_char import TokenizerChar


DEFAULT_TOKENIZER = '13a'


TOKENIZERS = {
    'none': BaseTokenizer,
    '13a': Tokenizer13a,
    'intl': TokenizerV14International,
    'zh': TokenizerZh,
    'ja-mecab': TokenizerJaMecab,
    'char': TokenizerChar,
}
