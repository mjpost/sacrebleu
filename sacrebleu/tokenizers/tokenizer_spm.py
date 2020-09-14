# -*- coding: utf-8 -*-

import os

from .tokenizer_none import NoneTokenizer

class TokenizerSPM(NoneTokenizer):
    def signature(self):
        return 'spm'

    def s3_get(self, bucket_name, s3_path, temp_file):
        """Pull a file directly from S3."""
        import boto3
        s3_resource = boto3.resource("s3")
        s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)

    def __init__(self):
        try:
            import sentencepiece as spm
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                '\n\nPlease install the sentencepiece library for SPM tokenization:'
                '\n\n  pip install sentencepiece '
            )
        self.sp = spm.SentencePieceProcessor()
        if not os.path.exists('sacrebleu_tokenizer_spm.model'):
            self.s3_get("fairusersglobal", "users/namangoyal/flores/spm_256000.model", open("sacrebleu_tokenizer_spm.model", 'wb'))
        self.sp.Load("sacrebleu_tokenizer_spm.model")

    def __call__(self, line):
        """Tokenizes all the characters in the input line.

        :param line: a segment to tokenize
        :return: the tokenized line
        """
        return " ".join(self.sp.EncodeAsPieces(line))
