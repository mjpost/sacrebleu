# -*- coding: utf-8 -*-

import os
import logging

from functools import lru_cache
from ..utils import SACREBLEU_DIR, download_file
from .tokenizer_base import BaseTokenizer

sacrelogger = logging.getLogger('sacrebleu')


SPM_MODELS = {
    "spm": {
        "url": "https://dl.fbaipublicfiles.com/fairseq/models/flores/sacrebleu_tokenizer_spm.model",
        "signature": "flores101",
    },
    # same as the default of "spm"
    "flores101": {
        "url": "https://dl.fbaipublicfiles.com/fairseq/models/flores/sacrebleu_tokenizer_spm.model",
        "signature": "flores101",
    },
    "flores200": {
        "url": "https://tinyurl.com/flores200sacrebleuspm",
        "signature": "flores200",
    },
    ### Added for spBLEU-1K tokenizer by AbdelRahim Elmadany
    "spBLEU-1K": {
          "url": "https://www.dlnlp.ai/spBLEU-1K/spbleu-1k_tokenizer_spm.model",
        "signature": "spBLEU-1K",
    },
}

class TokenizerSPM(BaseTokenizer):
    def signature(self):
        return self.name

    def __init__(self, key="spm"):
        self.name = SPM_MODELS[key]["signature"]

        if key == "spm":
            sacrelogger.warn("Tokenizer 'spm' has been changed to 'flores101', and may be removed in the future.")

        try:
            import sentencepiece as spm
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                '\n\nPlease install the sentencepiece library for SPM tokenization:'
                '\n\n  pip install sentencepiece '
            )
        self.sp = spm.SentencePieceProcessor()

        model_path = os.path.join(SACREBLEU_DIR, "models", os.path.basename(SPM_MODELS[key]["url"]))
        if not os.path.exists(model_path):
            url = SPM_MODELS[self.name]["url"]
            download_file(url, model_path)
        self.sp.Load(model_path)

    @lru_cache(maxsize=2**16)
    def __call__(self, line):
        """Tokenizes all the characters in the input line.

        :param line: a segment to tokenize
        :return: the tokenized line
        """
        return " ".join(self.sp.EncodeAsPieces(line))


class Flores200Tokenizer(TokenizerSPM):
    def __init__(self):
        super().__init__("flores200")

class Flores101Tokenizer(TokenizerSPM):
    def __init__(self):
        super().__init__("flores101")

### Added for spBLEU-1K tokenizer by AbdelRahim Elmadany
class spBLEU1KTokenizer(TokenizerSPM):
    def __init__(self):
        super().__init__("spBLEU-1K")