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

    def process_to_text(self, langpair=None):
        """Processes raw files to plain text files.

        :param langpair: The language pair to process. e.g. "en-de". If None, all files will be processed.
        """
        # ensure that the dataset is downloaded
        self.maybe_download()
        langpairs = self._get_langpair_metadata(langpair)

        for langpair in langpairs:

            corpus_dict = {}
            for meta in langpairs[langpair]:
                index, origin_file, field = self._split_index_and_filename(meta)

                origin_file = os.path.join(self._rawdir, origin_file)
                corpus_dict[field] = []

                with smart_open(origin_file) as fin:
                    for line in fin:
                        corpus_dict[field].append(line.rstrip("\n").split("\t")[index])

            src_lines = []
            src_docids = []

            prev_docid = None
            doc = ""
            for docid, sent in zip(corpus_dict["docid_src"], corpus_dict["src"]):
                if docid == prev_docid:
                    doc += sent
                elif prev_docid:
                    src_lines.append(doc)
                    src_docids.append(prev_docid)
                    doc = sent

                prev_docid = docid

            src_lines.append(doc)
            src_docids.append(docid)

            ref_lines = []
            ref_docids = []

            prev_docid = None
            doc = ""
            for docid, sent in zip(corpus_dict["docid_ref"], corpus_dict["ref"]):
                if docid == prev_docid:
                    doc += sent
                elif prev_docid:
                    ref_lines.append(doc)
                    ref_docids.append(prev_docid)
                    doc = sent

                prev_docid = docid

            ref_lines.append(doc)
            ref_docids.append(docid)

            src_output = self._get_txt_file_path(langpair, "src")
            ref_output = self._get_txt_file_path(langpair, "ref")
            src_docid_output = self._get_txt_file_path(langpair, "docid_src")
            ref_docid_output = self._get_txt_file_path(langpair, "docid_ref")

            with smart_open(src_output, "wt") as fout:
                for line in src_lines:
                    print(line, file=fout)

            with smart_open(ref_output, "wt") as fout:
                for line in ref_lines:
                    print(line, file=fout)

            with smart_open(src_docid_output, "wt") as fout:
                for line in src_docids:
                    print(line, file=fout)

            with smart_open(ref_docid_output, "wt") as fout:
                for line in ref_docids:
                    print(line, file=fout)