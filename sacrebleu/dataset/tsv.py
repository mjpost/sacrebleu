import os
from typing import List

from ..utils import smart_open
from .base import Dataset


class TSVDataset(Dataset):
    """
    The format used by the MTNT datasets. Data is in a single TSV file.
    """

    @staticmethod
    def _split_index_and_filename(meta):
        """
        Splits the index, filename and field from a metadata string.

        e.g. meta="3:en-de.tsv:src", -> (3, "en-de.tsv", "src")
        """
        arr = meta.split(":")
        assert len(arr) == 3, "Invalid metadata: {}".format(meta)
        index, filename, field = arr
        return int(index), filename, field

    def process_to_text(self, langpair=None):
        """Processes raw files to plain text files.

        :param langpair: The language pair to process. e.g. "en-de". If None, all files will be processed.
        """
        # ensure that the dataset is downloaded
        self.maybe_download()
        langpairs = self._get_langpair_metadata(langpair)

        for langpair in langpairs:
            for meta in langpairs[langpair]:
                index, origin_file, field = self._split_index_and_filename(meta)

                origin_file = os.path.join(self._rawdir, origin_file)
                output_file = self._get_txt_file_path(langpair, field)

                with smart_open(origin_file) as fin:
                    with smart_open(output_file, "wt") as fout:
                        for line in fin:
                            # be careful with empty source or reference lines
                            # MTNT2019/ja-en.final.tsv:632 `'1033\t718\t\t\n'`
                            print(line.rstrip("\n").split("\t")[index], file=fout)

    def fieldnames(self, langpair) -> List[str]:
        """Returns the field names of the dataset."""
        fields = []
        meta = self.langpairs[langpair]

        for item in meta:
            _, _, field = self._split_index_and_filename(item)
            fields.append(field)

        return fields


class WMTBiomedicalDataset(TSVDataset):
    """
    The format used by the WMT Biomedical datasets. Data is not aligned sent by sent.
    """

    def doc_align(self, langpair, sentences, field):
        """
        If the dataset is "doc aligned" instead of "sentence aligned",
        merge the sentences from the same document into a single line.

        :param: langpair: The language pair (e.g., "de-en")
        :param sentences: an iterable object, sentence level corpus.
        :param field: one of "src" and "ref".
        :return a list of merged docs.
        """
        # ensure that the dataset is downloaded
        self.maybe_download()
        langpairs = self._get_langpair_metadata(langpair)

        corpus_dict = {}
        for meta in langpairs[langpair]:
            index, origin_file, field_ = self._split_index_and_filename(meta)

            origin_file = os.path.join(self._rawdir, origin_file)
            corpus_dict[field_] = []

            with smart_open(origin_file) as fin:
                for line in fin:
                    corpus_dict[field_].append(line.rstrip("\n").split("\t")[index])

        docs = []
        docids = corpus_dict["docid_src"] if field == "src" else corpus_dict["docid_ref"]

        prev_docid = None
        doc = ""
        for docid, sent in zip(docids, sentences):
            if docid == prev_docid or not prev_docid:
                doc += sent
            elif prev_docid:
                docs.append(doc)
                doc = sent

            prev_docid = docid

        docs.append(doc)
        return docs
    
    @property
    def aligned_type(self):
        """
        Return the alignment type of the dataset.
        "sentence" or "documnet"
        """
        return "document"
