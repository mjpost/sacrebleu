from .tokenizer_zh import TokenizerZh


class TokenizerCJK(TokenizerZh):
    """Extend legacy Chinese tokenization with Hiragana and Katakana.

    This adds only U+3040--U+30FF to the ranges used by ``zh``; other
    characters and punctuation retain the legacy tokenizer's behavior.
    """

    def signature(self):
        return 'cjk'

    @staticmethod
    def _is_chinese_char(uchar):
        return '\u3040' <= uchar <= '\u30ff' or TokenizerZh._is_chinese_char(uchar)
