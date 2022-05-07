import os

from ..utils import smart_open
from .base import Dataset


class PlainTextDataset(Dataset):
    """
    The plain text format. Data is separated into source and reference files.
    Each line of the two files is aligned.
    """

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

            for field, origin_file in zip(fieldnames, origin_files):

                origin_file = os.path.join(self._rawdir, origin_file)
                output_file = self._get_txt_file_path(langpair, field)

                with smart_open(origin_file) as fin:
                    with smart_open(output_file, "wt") as fout:
                        for line in fin:
                            print(line.rstrip(), file=fout)

    def fieldnames(self, langpair):
        """
        Return a list of all the field names. For most source, this is just
        the source and the reference. For others, it might include the document
        ID for each line, or the original language (origLang).

        get_files() should return the same number of items as this.

        :param langpair: The language pair to get the field names for. e.g. "en-de".
        :return: A list of field names.
        """
        meta = self._get_langpair_metadata(langpair)
        length = max(meta.values(), key=len)
        if length == 1:
            return ["src"]
        else:
            return ["src", "ref"]


PLAIN_TEXT_DATASETS = {
    "wmt20/robust/set1": PlainTextDataset(
        "wmt20/robust/set1",
        data=["http://data.statmt.org/wmt20/robustness-task/robustness20-3-sets.zip"],
        description="WMT20 robustness task, set 1",
        langpairs={
            "en-ja": [
                "robustness20-3-sets/robustness20-set1-enja.en",
                "robustness20-3-sets/robustness20-set1-enja.ja",
            ],
            "en-de": [
                "robustness20-3-sets/robustness20-set1-ende.en",
                "robustness20-3-sets/robustness20-set1-ende.de",
            ],
        },
    ),
    "wmt20/robust/set2": PlainTextDataset(
        "wmt20/robust/set2",
        data=["http://data.statmt.org/wmt20/robustness-task/robustness20-3-sets.zip"],
        description="WMT20 robustness task, set 2",
        langpairs={
            "en-ja": [
                "robustness20-3-sets/robustness20-set2-enja.en",
                "robustness20-3-sets/robustness20-set2-enja.ja",
            ],
            "ja-en": [
                "robustness20-3-sets/robustness20-set2-jaen.ja",
                "robustness20-3-sets/robustness20-set2-jaen.en",
            ],
        },
    ),
    "wmt20/robust/set3": PlainTextDataset(
        "wmt20/robust/set3",
        data=["http://data.statmt.org/wmt20/robustness-task/robustness20-3-sets.zip"],
        description="WMT20 robustness task, set 3",
        langpairs={
            "de-en": [
                "robustness20-3-sets/robustness20-set3-deen.de",
                "robustness20-3-sets/robustness20-set3-deen.en",
            ],
        },
    ),
    "multi30k/2016": PlainTextDataset(
        "multi30k/2016",
        data=[
            "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/multi30k_test_sets_d3ec2a38.tar.gz"
        ],
        md5=["9cf8f22d57fee2ca2af3c682dfdc525b"],
        description="2016 flickr test set of Multi30k dataset",
        citation='@InProceedings{elliott-etal-2016-multi30k,\n    title = "{M}ulti30{K}: Multilingual {E}nglish-{G}erman Image Descriptions",\n    author = "Elliott, Desmond  and Frank, Stella  and Sima{\'}an, Khalil  and Specia, Lucia",\n    booktitle = "Proceedings of the 5th Workshop on Vision and Language",\n    month = aug,\n    year = "2016",\n    address = "Berlin, Germany",\n    publisher = "Association for Computational Linguistics",\n    url = "https://www.aclweb.org/anthology/W16-3210",\n    doi = "10.18653/v1/W16-3210",\n    pages = "70--74",\n}',
        langpairs={
            "en-fr": ["test_2016_flickr.en", "test_2016_flickr.fr"],
            "en-de": ["test_2016_flickr.en", "test_2016_flickr.de"],
            "en-cs": ["test_2016_flickr.en", "test_2016_flickr.cs"],
        },
    ),
    "multi30k/2017": PlainTextDataset(
        "multi30k/2017",
        data=[
            "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/multi30k_test_sets_d3ec2a38.tar.gz"
        ],
        md5=["9cf8f22d57fee2ca2af3c682dfdc525b"],
        description="2017 flickr test set of Multi30k dataset",
        citation='@InProceedings{elliott-etal-2016-multi30k,\n    title = "{M}ulti30{K}: Multilingual {E}nglish-{G}erman Image Descriptions",\n    author = "Elliott, Desmond  and Frank, Stella  and Sima{\'}an, Khalil  and Specia, Lucia",\n    booktitle = "Proceedings of the 5th Workshop on Vision and Language",\n    month = aug,\n    year = "2016",\n    address = "Berlin, Germany",\n    publisher = "Association for Computational Linguistics",\n    url = "https://www.aclweb.org/anthology/W16-3210",\n    doi = "10.18653/v1/W16-3210",\n    pages = "70--74",\n}\n\n@InProceedings{elliott-etal-2017-findings,\n    title = "Findings of the Second Shared Task on Multimodal Machine Translation and Multilingual Image Description",\n    author = {Elliott, Desmond  and Frank, Stella  and Barrault, Lo{\\"\\i}c  and Bougares, Fethi  and Specia, Lucia},\n    booktitle = "Proceedings of the Second Conference on Machine Translation",\n    month = sep,\n    year = "2017",\n    address = "Copenhagen, Denmark",\n    publisher = "Association for Computational Linguistics",\n    url = "https://www.aclweb.org/anthology/W17-4718",\n    doi = "10.18653/v1/W17-4718",\n    pages = "215--233",\n}\n',
        langpairs={
            "en-fr": ["test_2017_flickr.en", "test_2017_flickr.fr"],
            "en-de": ["test_2017_flickr.en", "test_2017_flickr.de"],
        },
    ),
    "multi30k/2018": PlainTextDataset(
        "multi30k/2018",
        data=[
            "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/multi30k_test_sets_d3ec2a38.tar.gz",
            "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2018_flickr.cs.gz",
            "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2018_flickr.de.gz",
            "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2018_flickr.fr.gz",
        ],
        md5=[
            "9cf8f22d57fee2ca2af3c682dfdc525b",
            "4c6b6490e58107b2e397c5e3e1690abc",
            "87e00327083dd69feaa029a8f7c1a047",
            "a64563e986438ed731a6713027c36bfd",
        ],
        description="2018 flickr test set of Multi30k dataset. See https://competitions.codalab.org/competitions/19917 for evaluation.",
        citation='@InProceedings{elliott-etal-2016-multi30k,\n    title = "{M}ulti30{K}: Multilingual {E}nglish-{G}erman Image Descriptions",\n    author = "Elliott, Desmond  and Frank, Stella  and Sima{\'}an, Khalil  and Specia, Lucia",\n    booktitle = "Proceedings of the 5th Workshop on Vision and Language",\n    month = aug,\n    year = "2016",\n    address = "Berlin, Germany",\n    publisher = "Association for Computational Linguistics",\n    url = "https://www.aclweb.org/anthology/W16-3210",\n    doi = "10.18653/v1/W16-3210",\n    pages = "70--74",\n}\n\n@InProceedings{barrault-etal-2018-findings,\n    title = "Findings of the Third Shared Task on Multimodal Machine Translation",\n    author = {Barrault, Lo{\\"\\i}c  and Bougares, Fethi  and Specia, Lucia  and Lala, Chiraag  and Elliott, Desmond  and Frank, Stella},\n    booktitle = "Proceedings of the Third Conference on Machine Translation: Shared Task Papers",\n    month = oct,\n    year = "2018",\n    address = "Belgium, Brussels",\n    publisher = "Association for Computational Linguistics",\n    url = "https://www.aclweb.org/anthology/W18-6402",\n    doi = "10.18653/v1/W18-6402",\n    pages = "304--323",\n}\n',
        langpairs={
            "en-fr": ["test_2018_flickr.en", "test_2018_flickr.fr"],
            "en-de": ["test_2018_flickr.en", "test_2018_flickr.de"],
            "en-cs": ["test_2018_flickr.en", "test_2018_flickr.cs"],
        },
    ),
}
