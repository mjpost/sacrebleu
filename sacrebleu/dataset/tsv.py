from .base import Dataset


class TSVDataset(Dataset):
    """
    The format used by the MTNT datasets. Data is in a single TSV file.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_to_text(self):
        """
        Class method that essentially does what utils/process_to_text() does.

        This should be implemented by subclasses. Note: process_to_text should write the
        fields in a different format: ~/.sacrebleu/DATASET/DATASET.LANGPAIR.FIELDNAME
        (instead of the current ~/.sacrebleu/DATASET/LANGPAIR.{SRC,REF})
        """
        pass

    def fieldnames(self):
        """
        Return a list of all the field names. For most source, this is just
        the source and the reference. For others, it might include the document
        ID for each line, or the original language (origLang).

        get_files() should return the same number of items as this.
        """
        pass

    def __iter__(self):
        """
        Iterates over all fields (source, references, and other metadata) defined
        by the dataset.
        """
        pass

    def source(self):
        """
        Return an iterable over the source lines.
        """
        pass

    def references(self):
        """
        Return an iterable over the references.
        """
        pass

    def get_source_file(self):
        pass

    def get_files(self):
        pass


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
    "mtnt1.1/train": TSVDataset(
        "mtnt1.1/train",
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
        description="Training data for the Machine Translation of Noisy Text task: http://www.cs.cmu.edu/~pmichel1/mtnt/",
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
