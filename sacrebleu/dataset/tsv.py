import os

from ..utils import smart_open
from .base import Dataset


class TSVDataset(Dataset):
    """
    The format used by the MTNT datasets. Data is in a single TSV file.
    """

    @staticmethod
    def _split_index_and_filename(meta, field):
        """
        Splits the index and filename from a metadata string.

        e.g. meta="3:en-de.tsv", filed=[Any value] -> (3, "en-de.tsv")
             "en-de.tsv", filed="src" -> (1, "en-de.tsv")
             "en-de.tsv", filed="tgt" -> (2, "en-de.tsv")
        """
        arr = meta.split(":")
        if len(arr) == 2:
            try:
                index = int(arr[0])
            except ValueError:
                raise Exception(f"Invalid meta for TSVDataset: {meta}")
            return index, arr[1]

        else:
            index = 0 if field == "src" else 1
            return index, meta

    def process_to_text(self, langpair=None):
        """Processes raw files to plain text files.

        :param langpair: The language pair to process. e.g. "en-de". If None, all files will be processed.
        """
        # ensure that the dataset is downloaded
        self.maybe_download()
        langpairs = self._get_langpair_metadata(langpair)

        for langpair in langpairs:
            fieldnames = self.fieldnames(langpair)
            origin_files = [
                os.path.join(self._rawdir, path) for path in langpairs[langpair]
            ]

            for field, origin_file, meta in zip(
                fieldnames, origin_files, langpairs[langpair]
            ):
                index, origin_file = self._split_index_and_filename(meta, field)

                origin_file = os.path.join(self._rawdir, origin_file)
                output_file = self._get_txt_file_path(langpair, field)

                with smart_open(origin_file) as fin:
                    with smart_open(output_file, "wt") as fout:
                        for line in fin:
                            # be careful with empty source or reference lines
                            # MTNT2019/ja-en.final.tsv:632 `'1033\t718\t\t\n'`
                            print(line.rstrip("\n").split("\t")[index], file=fout)


TSV_DATASETS = {
    "mtnt2019": TSVDataset(
        "mtnt2019",
        data=["https://pmichel31415.github.io/hosting/MTNT2019.tar.gz"],
        description="Test set for the WMT 19 robustness shared task",
        md5=["78a672e1931f106a8549023c0e8af8f6"],
        langpairs={
            "en-fr": ["2:MTNT2019/en-fr.final.tsv", "3:MTNT2019/en-fr.final.tsv"],
            "fr-en": ["2:MTNT2019/fr-en.final.tsv", "3:MTNT2019/fr-en.final.tsv"],
            "en-ja": ["2:MTNT2019/en-ja.final.tsv", "3:MTNT2019/en-ja.final.tsv"],
            "ja-en": ["2:MTNT2019/ja-en.final.tsv", "3:MTNT2019/ja-en.final.tsv"],
        },
    ),
    "mtnt1.1/test": TSVDataset(
        "mtnt1.1/test",
        data=[
            "https://github.com/pmichel31415/mtnt/releases/download/v1.1/MTNT.1.1.tar.gz"
        ],
        description="Test data for the Machine Translation of Noisy Text task: http://www.cs.cmu.edu/~pmichel1/mtnt/",
        citation='@InProceedings{michel2018a:mtnt,\n    author = "Michel, Paul and Neubig, Graham",\n    title = "MTNT: A Testbed for Machine Translation of Noisy Text",\n    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",\n    year = "2018",\n    publisher = "Association for Computational Linguistics",\n    pages = "543--553",\n    location = "Brussels, Belgium",\n    url = "http://aclweb.org/anthology/D18-1050"\n}',
        md5=["8ce1831ac584979ba8cdcd9d4be43e1d"],
        langpairs={
            "en-fr": ["1:MTNT/test/test.en-fr.tsv", "2:MTNT/test/test.en-fr.tsv"],
            "fr-en": ["1:MTNT/test/test.fr-en.tsv", "2:MTNT/test/test.fr-en.tsv"],
            "en-ja": ["1:MTNT/test/test.en-ja.tsv", "2:MTNT/test/test.en-ja.tsv"],
            "ja-en": ["1:MTNT/test/test.ja-en.tsv", "2:MTNT/test/test.ja-en.tsv"],
        },
    ),
    "mtnt1.1/valid": TSVDataset(
        "mtnt1.1/valid",
        data=[
            "https://github.com/pmichel31415/mtnt/releases/download/v1.1/MTNT.1.1.tar.gz"
        ],
        description="Validation data for the Machine Translation of Noisy Text task: http://www.cs.cmu.edu/~pmichel1/mtnt/",
        citation='@InProceedings{michel2018a:mtnt,\n    author = "Michel, Paul and Neubig, Graham",\n    title = "MTNT: A Testbed for Machine Translation of Noisy Text",\n    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",\n    year = "2018",\n    publisher = "Association for Computational Linguistics",\n    pages = "543--553",\n    location = "Brussels, Belgium",\n    url = "http://aclweb.org/anthology/D18-1050"\n}',
        md5=["8ce1831ac584979ba8cdcd9d4be43e1d"],
        langpairs={
            "en-fr": ["1:MTNT/valid/valid.en-fr.tsv", "2:MTNT/valid/valid.en-fr.tsv"],
            "fr-en": ["1:MTNT/valid/valid.fr-en.tsv", "2:MTNT/valid/valid.fr-en.tsv"],
            "en-ja": ["1:MTNT/valid/valid.en-ja.tsv", "2:MTNT/valid/valid.en-ja.tsv"],
            "ja-en": ["1:MTNT/valid/valid.ja-en.tsv", "2:MTNT/valid/valid.ja-en.tsv"],
        },
    ),
    "mtnt1.1/train": TSVDataset(
        "mtnt1.1/train",
        data=[
            "https://github.com/pmichel31415/mtnt/releases/download/v1.1/MTNT.1.1.tar.gz"
        ],
        description="Validation data for the Machine Translation of Noisy Text task: http://www.cs.cmu.edu/~pmichel1/mtnt/",
        citation='@InProceedings{michel2018a:mtnt,\n    author = "Michel, Paul and Neubig, Graham",\n    title = "MTNT: A Testbed for Machine Translation of Noisy Text",\n    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",\n    year = "2018",\n    publisher = "Association for Computational Linguistics",\n    pages = "543--553",\n    location = "Brussels, Belgium",\n    url = "http://aclweb.org/anthology/D18-1050"\n}',
        md5=["8ce1831ac584979ba8cdcd9d4be43e1d"],
        langpairs={
            "en-fr": ["1:MTNT/train/train.en-fr.tsv", "2:MTNT/train/train.en-fr.tsv"],
            "fr-en": ["1:MTNT/train/train.fr-en.tsv", "2:MTNT/train/train.fr-en.tsv"],
            "en-ja": ["1:MTNT/train/train.en-ja.tsv", "2:MTNT/train/train.en-ja.tsv"],
            "ja-en": ["1:MTNT/train/train.ja-en.tsv", "2:MTNT/train/train.ja-en.tsv"],
        },
    ),
}
