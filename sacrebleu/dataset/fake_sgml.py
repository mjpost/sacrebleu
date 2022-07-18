import os
import re

from ..utils import smart_open
from .base import Dataset


class FakeSGMLDataset(Dataset):
    """
    The fake SGML format used by WMT prior to 2021. Can't be properly parsed.
    Source and reference(s) in separate files.
    """

    def _convert_format(self, input_file_path, output_filep_path):
        """
        Extract data from raw file and convert to raw txt format.
        """
        with smart_open(input_file_path) as fin, smart_open(
            output_filep_path, "wt"
        ) as fout:
            for line in fin:
                if line.startswith("<seg "):
                    line = self._clean(re.sub(r"<seg.*?>(.*)</seg>.*?", "\\1", line))
                    print(line, file=fout)

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
                self._convert_format(origin_file, output_file)

    def fieldnames(self, langpair):
        """
        Return a list of all the field names. For most source, this is just
        the source and the reference. For others, it might include the document
        ID for each line, or the original language (origLang).

        get_files() should return the same number of items as this.

        TODO genre, docid
        """
        meta = self._get_langpair_metadata(langpair)
        length = len(meta[langpair])

        assert (
            length >= 2
        ), f"Each language pair in {self.name} must have at least 2 fileds."

        if length == 2:
            return ["src", "ref"]
        else:
            fields = ["src"]
            for i, _ in enumerate(meta[langpair][1:]):
                fields.append(f"ref:{i}")

            return fields


class WMTAdditionDataset(FakeSGMLDataset):
    """
    Handle special case of WMT Google addition dataset.
    """

    def _convert_format(self, input_file_path, output_filep_path):
        if input_file_path.endswith(".sgm"):
            return super()._convert_format(input_file_path, output_filep_path)
        else:
            with smart_open(input_file_path) as fin:
                with smart_open(output_filep_path, "wt") as fout:
                    for line in fin:
                        print(line.rstrip(), file=fout)


FAKE_SGML_DATASETS = {
    "wmt20/tworefs": FakeSGMLDataset(
        "wmt20/tworefs",
        data=["http://data.statmt.org/wmt20/translation-task/test.tgz"],
        description="WMT20 news test sets with two references",
        md5=["3b1f777cfd2fb15ccf66e9bfdb2b1699"],
        langpairs={
            "de-en": [
                "sgm/newstest2020-deen-src.de.sgm",
                "sgm/newstest2020-deen-ref.en.sgm",
                "sgm/newstestB2020-deen-ref.en.sgm",
            ],
            "en-de": [
                "sgm/newstest2020-ende-src.en.sgm",
                "sgm/newstest2020-ende-ref.de.sgm",
                "sgm/newstestB2020-ende-ref.de.sgm",
            ],
            "en-zh": [
                "sgm/newstest2020-enzh-src.en.sgm",
                "sgm/newstest2020-enzh-ref.zh.sgm",
                "sgm/newstestB2020-enzh-ref.zh.sgm",
            ],
            "ru-en": [
                "sgm/newstest2020-ruen-src.ru.sgm",
                "sgm/newstest2020-ruen-ref.en.sgm",
                "sgm/newstestB2020-ruen-ref.en.sgm",
            ],
            "zh-en": [
                "sgm/newstest2020-zhen-src.zh.sgm",
                "sgm/newstest2020-zhen-ref.en.sgm",
                "sgm/newstestB2020-zhen-ref.en.sgm",
            ],
        },
    ),
    "wmt20": FakeSGMLDataset(
        "wmt20",
        data=["http://data.statmt.org/wmt20/translation-task/test.tgz"],
        description="Official evaluation data for WMT20",
        md5=["3b1f777cfd2fb15ccf66e9bfdb2b1699"],
        langpairs={
            "cs-en": [
                "sgm/newstest2020-csen-src.cs.sgm",
                "sgm/newstest2020-csen-ref.en.sgm",
            ],
            "de-en": [
                "sgm/newstest2020-deen-src.de.sgm",
                "sgm/newstest2020-deen-ref.en.sgm",
            ],
            "de-fr": [
                "sgm/newstest2020-defr-src.de.sgm",
                "sgm/newstest2020-defr-ref.fr.sgm",
            ],
            "en-cs": [
                "sgm/newstest2020-encs-src.en.sgm",
                "sgm/newstest2020-encs-ref.cs.sgm",
            ],
            "en-de": [
                "sgm/newstest2020-ende-src.en.sgm",
                "sgm/newstest2020-ende-ref.de.sgm",
            ],
            "en-iu": [
                "sgm/newstest2020-eniu-src.en.sgm",
                "sgm/newstest2020-eniu-ref.iu.sgm",
            ],
            "en-ja": [
                "sgm/newstest2020-enja-src.en.sgm",
                "sgm/newstest2020-enja-ref.ja.sgm",
            ],
            "en-km": [
                "sgm/newstest2020-enkm-src.en.sgm",
                "sgm/newstest2020-enkm-ref.km.sgm",
            ],
            "en-pl": [
                "sgm/newstest2020-enpl-src.en.sgm",
                "sgm/newstest2020-enpl-ref.pl.sgm",
            ],
            "en-ps": [
                "sgm/newstest2020-enps-src.en.sgm",
                "sgm/newstest2020-enps-ref.ps.sgm",
            ],
            "en-ru": [
                "sgm/newstest2020-enru-src.en.sgm",
                "sgm/newstest2020-enru-ref.ru.sgm",
            ],
            "en-ta": [
                "sgm/newstest2020-enta-src.en.sgm",
                "sgm/newstest2020-enta-ref.ta.sgm",
            ],
            "en-zh": [
                "sgm/newstest2020-enzh-src.en.sgm",
                "sgm/newstest2020-enzh-ref.zh.sgm",
            ],
            "fr-de": [
                "sgm/newstest2020-frde-src.fr.sgm",
                "sgm/newstest2020-frde-ref.de.sgm",
            ],
            "iu-en": [
                "sgm/newstest2020-iuen-src.iu.sgm",
                "sgm/newstest2020-iuen-ref.en.sgm",
            ],
            "ja-en": [
                "sgm/newstest2020-jaen-src.ja.sgm",
                "sgm/newstest2020-jaen-ref.en.sgm",
            ],
            "km-en": [
                "sgm/newstest2020-kmen-src.km.sgm",
                "sgm/newstest2020-kmen-ref.en.sgm",
            ],
            "pl-en": [
                "sgm/newstest2020-plen-src.pl.sgm",
                "sgm/newstest2020-plen-ref.en.sgm",
            ],
            "ps-en": [
                "sgm/newstest2020-psen-src.ps.sgm",
                "sgm/newstest2020-psen-ref.en.sgm",
            ],
            "ru-en": [
                "sgm/newstest2020-ruen-src.ru.sgm",
                "sgm/newstest2020-ruen-ref.en.sgm",
            ],
            "ta-en": [
                "sgm/newstest2020-taen-src.ta.sgm",
                "sgm/newstest2020-taen-ref.en.sgm",
            ],
            "zh-en": [
                "sgm/newstest2020-zhen-src.zh.sgm",
                "sgm/newstest2020-zhen-ref.en.sgm",
            ],
        },
    ),
    "wmt20/dev": FakeSGMLDataset(
        "wmt20/dev",
        data=["http://data.statmt.org/wmt20/translation-task/dev.tgz"],
        description="Development data for tasks new to 2020.",
        md5=["037f2b37aab74febbb1b2307dc2afb54"],
        langpairs={
            "iu-en": [
                "dev/newsdev2020-iuen-src.iu.sgm",
                "dev/newsdev2020-iuen-ref.en.sgm",
            ],
            "en-iu": [
                "dev/newsdev2020-eniu-src.en.sgm",
                "dev/newsdev2020-eniu-ref.iu.sgm",
            ],
            "ja-en": [
                "dev/newsdev2020-jaen-src.ja.sgm",
                "dev/newsdev2020-jaen-ref.en.sgm",
            ],
            "en-ja": [
                "dev/newsdev2020-enja-src.en.sgm",
                "dev/newsdev2020-enja-ref.ja.sgm",
            ],
            "pl-en": [
                "dev/newsdev2020-plen-src.pl.sgm",
                "dev/newsdev2020-plen-ref.en.sgm",
            ],
            "en-pl": [
                "dev/newsdev2020-enpl-src.en.sgm",
                "dev/newsdev2020-enpl-ref.pl.sgm",
            ],
            "ta-en": [
                "dev/newsdev2020-taen-src.ta.sgm",
                "dev/newsdev2020-taen-ref.en.sgm",
            ],
            "en-ta": [
                "dev/newsdev2020-enta-src.en.sgm",
                "dev/newsdev2020-enta-ref.ta.sgm",
            ],
        },
    ),
    "wmt19": FakeSGMLDataset(
        "wmt19",
        data=["http://data.statmt.org/wmt19/translation-task/test.tgz"],
        description="Official evaluation data.",
        md5=["84de7162d158e28403103b01aeefc39a"],
        citation=r"""@proceedings{ws-2019-machine,
    title = "Proceedings of the Fourth Conference on Machine Translation (Volume 1: Research Papers)",
    editor = "Bojar, Ond{\v{r}}ej  and
      Chatterjee, Rajen  and
      Federmann, Christian  and
      Fishel, Mark  and
      Graham, Yvette  and
      Haddow, Barry  and
      Huck, Matthias  and
      Yepes, Antonio Jimeno  and
      Koehn, Philipp  and
      Martins, Andr{\'e}  and
      Monz, Christof  and
      Negri, Matteo  and
      N{\'e}v{\'e}ol, Aur{\'e}lie  and
      Neves, Mariana  and
      Post, Matt  and
      Turchi, Marco  and
      Verspoor, Karin",
    month = aug,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W19-5200",
}""",
        langpairs={
            "cs-de": [
                "sgm/newstest2019-csde-src.cs.sgm",
                "sgm/newstest2019-csde-ref.de.sgm",
            ],
            "de-cs": [
                "sgm/newstest2019-decs-src.de.sgm",
                "sgm/newstest2019-decs-ref.cs.sgm",
            ],
            "de-en": [
                "sgm/newstest2019-deen-src.de.sgm",
                "sgm/newstest2019-deen-ref.en.sgm",
            ],
            "de-fr": [
                "sgm/newstest2019-defr-src.de.sgm",
                "sgm/newstest2019-defr-ref.fr.sgm",
            ],
            "en-cs": [
                "sgm/newstest2019-encs-src.en.sgm",
                "sgm/newstest2019-encs-ref.cs.sgm",
            ],
            "en-de": [
                "sgm/newstest2019-ende-src.en.sgm",
                "sgm/newstest2019-ende-ref.de.sgm",
            ],
            "en-fi": [
                "sgm/newstest2019-enfi-src.en.sgm",
                "sgm/newstest2019-enfi-ref.fi.sgm",
            ],
            "en-gu": [
                "sgm/newstest2019-engu-src.en.sgm",
                "sgm/newstest2019-engu-ref.gu.sgm",
            ],
            "en-kk": [
                "sgm/newstest2019-enkk-src.en.sgm",
                "sgm/newstest2019-enkk-ref.kk.sgm",
            ],
            "en-lt": [
                "sgm/newstest2019-enlt-src.en.sgm",
                "sgm/newstest2019-enlt-ref.lt.sgm",
            ],
            "en-ru": [
                "sgm/newstest2019-enru-src.en.sgm",
                "sgm/newstest2019-enru-ref.ru.sgm",
            ],
            "en-zh": [
                "sgm/newstest2019-enzh-src.en.sgm",
                "sgm/newstest2019-enzh-ref.zh.sgm",
            ],
            "fi-en": [
                "sgm/newstest2019-fien-src.fi.sgm",
                "sgm/newstest2019-fien-ref.en.sgm",
            ],
            "fr-de": [
                "sgm/newstest2019-frde-src.fr.sgm",
                "sgm/newstest2019-frde-ref.de.sgm",
            ],
            "gu-en": [
                "sgm/newstest2019-guen-src.gu.sgm",
                "sgm/newstest2019-guen-ref.en.sgm",
            ],
            "kk-en": [
                "sgm/newstest2019-kken-src.kk.sgm",
                "sgm/newstest2019-kken-ref.en.sgm",
            ],
            "lt-en": [
                "sgm/newstest2019-lten-src.lt.sgm",
                "sgm/newstest2019-lten-ref.en.sgm",
            ],
            "ru-en": [
                "sgm/newstest2019-ruen-src.ru.sgm",
                "sgm/newstest2019-ruen-ref.en.sgm",
            ],
            "zh-en": [
                "sgm/newstest2019-zhen-src.zh.sgm",
                "sgm/newstest2019-zhen-ref.en.sgm",
            ],
        },
    ),
    "wmt19/dev": FakeSGMLDataset(
        "wmt19/dev",
        data=["http://data.statmt.org/wmt19/translation-task/dev.tgz"],
        description="Development data for tasks new to 2019.",
        md5=["f2ec7af5947c19e0cacb3882eb208002"],
        langpairs={
            "lt-en": [
                "dev/newsdev2019-lten-src.lt.sgm",
                "dev/newsdev2019-lten-ref.en.sgm",
            ],
            "en-lt": [
                "dev/newsdev2019-enlt-src.en.sgm",
                "dev/newsdev2019-enlt-ref.lt.sgm",
            ],
            "gu-en": [
                "dev/newsdev2019-guen-src.gu.sgm",
                "dev/newsdev2019-guen-ref.en.sgm",
            ],
            "en-gu": [
                "dev/newsdev2019-engu-src.en.sgm",
                "dev/newsdev2019-engu-ref.gu.sgm",
            ],
            "kk-en": [
                "dev/newsdev2019-kken-src.kk.sgm",
                "dev/newsdev2019-kken-ref.en.sgm",
            ],
            "en-kk": [
                "dev/newsdev2019-enkk-src.en.sgm",
                "dev/newsdev2019-enkk-ref.kk.sgm",
            ],
        },
    ),
    "wmt19/google/ar": WMTAdditionDataset(
        "wmt19/google/ar",
        data=[
            "http://data.statmt.org/wmt19/translation-task/test.tgz",
            "https://raw.githubusercontent.com/google/wmt19-paraphrased-references/master/wmt19/ende/wmt19-ende-ar.ref",
        ],
        description="Additional high-quality reference for WMT19/en-de.",
        md5=["84de7162d158e28403103b01aeefc39a", "d66d9e91548ced0ac476f2390e32e2de"],
        citation="@misc{freitag2020bleu,\n    title={{BLEU} might be Guilty but References are not Innocent},\n    author={Markus Freitag and David Grangier and Isaac Caswell},\n    year={2020},\n    eprint={2004.06063},\n    archivePrefix={arXiv},\n    primaryClass={cs.CL}",
        langpairs={
            "en-de": ["sgm/newstest2019-ende-src.en.sgm", "wmt19-ende-ar.ref"],
        },
    ),
    "wmt19/google/arp": WMTAdditionDataset(
        "wmt19/google/arp",
        data=[
            "http://data.statmt.org/wmt19/translation-task/test.tgz",
            "https://raw.githubusercontent.com/google/wmt19-paraphrased-references/master/wmt19/ende/wmt19-ende-arp.ref",
        ],
        description="Additional paraphrase of wmt19/google/ar.",
        md5=["84de7162d158e28403103b01aeefc39a", "c70ea808cf2bff621ad7a8fddd4deca9"],
        citation="@misc{freitag2020bleu,\n    title={{BLEU} might be Guilty but References are not Innocent},\n    author={Markus Freitag and David Grangier and Isaac Caswell},\n    year={2020},\n    eprint={2004.06063},\n    archivePrefix={arXiv},\n    primaryClass={cs.CL}",
        langpairs={
            "en-de": ["sgm/newstest2019-ende-src.en.sgm", "wmt19-ende-arp.ref"],
        },
    ),
    "wmt19/google/wmtp": WMTAdditionDataset(
        "wmt19/google/wmtp",
        data=[
            "http://data.statmt.org/wmt19/translation-task/test.tgz",
            "https://raw.githubusercontent.com/google/wmt19-paraphrased-references/master/wmt19/ende/wmt19-ende-wmtp.ref",
        ],
        description="Additional paraphrase of the official WMT19 reference.",
        md5=["84de7162d158e28403103b01aeefc39a", "587c660ee5fd44727f0db025b71c6a82"],
        citation="@misc{freitag2020bleu,\n    title={{BLEU} might be Guilty but References are not Innocent},\n    author={Markus Freitag and David Grangier and Isaac Caswell},\n    year={2020},\n    eprint={2004.06063},\n    archivePrefix={arXiv},\n    primaryClass={cs.CL}",
        langpairs={
            "en-de": ["sgm/newstest2019-ende-src.en.sgm", "wmt19-ende-wmtp.ref"],
        },
    ),
    "wmt19/google/hqr": WMTAdditionDataset(
        "wmt19/google/hqr",
        data=[
            "http://data.statmt.org/wmt19/translation-task/test.tgz",
            "https://raw.githubusercontent.com/google/wmt19-paraphrased-references/master/wmt19/ende/wmt19-ende-hqr.ref",
        ],
        description="Best human selected-reference between wmt19 and wmt19/google/ar.",
        md5=["84de7162d158e28403103b01aeefc39a", "d9221135f62d7152de041f5bfc8efaea"],
        citation="@misc{freitag2020bleu,\n    title={{BLEU} might be Guilty but References are not Innocent},\n    author={Markus Freitag and David Grangier and Isaac Caswell},\n    year={2020},\n    eprint={2004.06063},\n    archivePrefix={arXiv},\n    primaryClass={cs.CL}",
        langpairs={
            "en-de": ["sgm/newstest2019-ende-src.en.sgm", "wmt19-ende-hqr.ref"],
        },
    ),
    "wmt19/google/hqp": WMTAdditionDataset(
        "wmt19/google/hqp",
        data=[
            "http://data.statmt.org/wmt19/translation-task/test.tgz",
            "https://raw.githubusercontent.com/google/wmt19-paraphrased-references/master/wmt19/ende/wmt19-ende-hqp.ref",
        ],
        description="Best human-selected reference between wmt19/google/arp and wmt19/google/wmtp.",
        md5=["84de7162d158e28403103b01aeefc39a", "b7c3a07a59c8eccea5367e9ec5417a8a"],
        citation="@misc{freitag2020bleu,\n    title={{BLEU} might be Guilty but References are not Innocent},\n    author={Markus Freitag and David Grangier and Isaac Caswell},\n    year={2020},\n    eprint={2004.06063},\n    archivePrefix={arXiv},\n    primaryClass={cs.CL}",
        langpairs={
            "en-de": ["sgm/newstest2019-ende-src.en.sgm", "wmt19-ende-hqp.ref"],
        },
    ),
    "wmt19/google/hqall": WMTAdditionDataset(
        "wmt19/google/hqall",
        data=[
            "http://data.statmt.org/wmt19/translation-task/test.tgz",
            "https://raw.githubusercontent.com/google/wmt19-paraphrased-references/master/wmt19/ende/wmt19-ende-hqall.ref",
        ],
        description="Best human-selected reference among original official reference and the Google reference and paraphrases.",
        md5=["84de7162d158e28403103b01aeefc39a", "edecf10ced59e10b703a6fbcf1fa9dfa"],
        citation="@misc{freitag2020bleu,\n    title={{BLEU} might be Guilty but References are not Innocent},\n    author={Markus Freitag and David Grangier and Isaac Caswell},\n    year={2020},\n    eprint={2004.06063},\n    archivePrefix={arXiv},\n    primaryClass={cs.CL}",
        langpairs={
            "en-de": ["sgm/newstest2019-ende-src.en.sgm", "wmt19-ende-hqall.ref"],
        },
    ),
    "wmt18": FakeSGMLDataset(
        "wmt18",
        data=["http://data.statmt.org/wmt18/translation-task/test.tgz"],
        md5=["f996c245ecffea23d0006fa4c34e9064"],
        description="Official evaluation data.",
        citation='@inproceedings{bojar-etal-2018-findings,\n    title = "Findings of the 2018 Conference on Machine Translation ({WMT}18)",\n    author = "Bojar, Ond{\v{r}}ej  and\n      Federmann, Christian  and\n      Fishel, Mark  and\n      Graham, Yvette  and\n      Haddow, Barry  and\n      Koehn, Philipp  and\n      Monz, Christof",\n    booktitle = "Proceedings of the Third Conference on Machine Translation: Shared Task Papers",\n    month = oct,\n    year = "2018",\n    address = "Belgium, Brussels",\n    publisher = "Association for Computational Linguistics",\n    url = "https://www.aclweb.org/anthology/W18-6401",\n    pages = "272--303",\n}',
        langpairs={
            "cs-en": [
                "test/newstest2018-csen-src.cs.sgm",
                "test/newstest2018-csen-ref.en.sgm",
            ],
            "de-en": [
                "test/newstest2018-deen-src.de.sgm",
                "test/newstest2018-deen-ref.en.sgm",
            ],
            "en-cs": [
                "test/newstest2018-encs-src.en.sgm",
                "test/newstest2018-encs-ref.cs.sgm",
            ],
            "en-de": [
                "test/newstest2018-ende-src.en.sgm",
                "test/newstest2018-ende-ref.de.sgm",
            ],
            "en-et": [
                "test/newstest2018-enet-src.en.sgm",
                "test/newstest2018-enet-ref.et.sgm",
            ],
            "en-fi": [
                "test/newstest2018-enfi-src.en.sgm",
                "test/newstest2018-enfi-ref.fi.sgm",
            ],
            "en-ru": [
                "test/newstest2018-enru-src.en.sgm",
                "test/newstest2018-enru-ref.ru.sgm",
            ],
            "et-en": [
                "test/newstest2018-eten-src.et.sgm",
                "test/newstest2018-eten-ref.en.sgm",
            ],
            "fi-en": [
                "test/newstest2018-fien-src.fi.sgm",
                "test/newstest2018-fien-ref.en.sgm",
            ],
            "ru-en": [
                "test/newstest2018-ruen-src.ru.sgm",
                "test/newstest2018-ruen-ref.en.sgm",
            ],
            "en-tr": [
                "test/newstest2018-entr-src.en.sgm",
                "test/newstest2018-entr-ref.tr.sgm",
            ],
            "tr-en": [
                "test/newstest2018-tren-src.tr.sgm",
                "test/newstest2018-tren-ref.en.sgm",
            ],
            "en-zh": [
                "test/newstest2018-enzh-src.en.sgm",
                "test/newstest2018-enzh-ref.zh.sgm",
            ],
            "zh-en": [
                "test/newstest2018-zhen-src.zh.sgm",
                "test/newstest2018-zhen-ref.en.sgm",
            ],
        },
    ),
    "wmt18/test-ts": FakeSGMLDataset(
        "wmt18/test-ts",
        data=["http://data.statmt.org/wmt18/translation-task/test-ts.tgz"],
        md5=["5c621a34d512cc2dd74162ae7d00b320"],
        description="Official evaluation sources with extra test sets interleaved.",
        langpairs={
            "cs-en": ["test-ts/newstest2018-csen-src-ts.cs.sgm", "test-ts/newstest2018-csen-ref-ts.en.sgm"],
            "de-en": ["test-ts/newstest2018-deen-src-ts.de.sgm", "test-ts/newstest2018-deen-ref-ts.en.sgm"],
            "en-cs": ["test-ts/newstest2018-encs-src-ts.en.sgm", "test-ts/newstest2018-encs-ref-ts.cs.sgm"],
            "en-de": ["test-ts/newstest2018-ende-src-ts.en.sgm", "test-ts/newstest2018-ende-ref-ts.de.sgm"],
            "en-et": ["test-ts/newstest2018-enet-src-ts.en.sgm", "test-ts/newstest2018-enet-ref-ts.et.sgm"],
            "en-fi": ["test-ts/newstest2018-enfi-src-ts.en.sgm", "test-ts/newstest2018-enfi-ref-ts.fi.sgm"],
            "en-ru": ["test-ts/newstest2018-enru-src-ts.en.sgm", "test-ts/newstest2018-enru-ref-ts.ru.sgm"],
            "et-en": ["test-ts/newstest2018-eten-src-ts.et.sgm", "test-ts/newstest2018-eten-ref-ts.en.sgm"],
            "fi-en": ["test-ts/newstest2018-fien-src-ts.fi.sgm", "test-ts/newstest2018-fien-ref-ts.en.sgm"],
            "ru-en": ["test-ts/newstest2018-ruen-src-ts.ru.sgm", "test-ts/newstest2018-ruen-ref-ts.en.sgm"],
            "en-tr": ["test-ts/newstest2018-entr-src-ts.en.sgm", "test-ts/newstest2018-entr-ref-ts.tr.sgm"],
            "tr-en": ["test-ts/newstest2018-tren-src-ts.tr.sgm", "test-ts/newstest2018-tren-ref-ts.en.sgm"],
            "en-zh": ["test-ts/newstest2018-enzh-src-ts.en.sgm", "test-ts/newstest2018-enzh-ref-ts.zh.sgm"],
            "zh-en": ["test-ts/newstest2018-zhen-src-ts.zh.sgm", "test-ts/newstest2018-zhen-ref-ts.en.sgm"],
        },
    ),
    "wmt18/dev": FakeSGMLDataset(
        "wmt18/dev",
        data=["http://data.statmt.org/wmt18/translation-task/dev.tgz"],
        md5=["486f391da54a7a3247f02ebd25996f24"],
        description="Development data (Estonian<>English).",
        langpairs={
            "et-en": [
                "dev/newsdev2018-eten-src.et.sgm",
                "dev/newsdev2018-eten-ref.en.sgm",
            ],
            "en-et": [
                "dev/newsdev2018-enet-src.en.sgm",
                "dev/newsdev2018-enet-ref.et.sgm",
            ],
        },
    ),
    "wmt17": FakeSGMLDataset(
        "wmt17",
        data=["http://data.statmt.org/wmt17/translation-task/test.tgz"],
        md5=["86a1724c276004aa25455ae2a04cef26"],
        description="Official evaluation data.",
        citation="@InProceedings{bojar-EtAl:2017:WMT1,\n  author    = {Bojar, Ond\\v{r}ej  and  Chatterjee, Rajen  and  Federmann, Christian  and  Graham, Yvette  and  Haddow, Barry  and  Huang, Shujian  and  Huck, Matthias  and  Koehn, Philipp  and  Liu, Qun  and  Logacheva, Varvara  and  Monz, Christof  and  Negri, Matteo  and  Post, Matt  and  Rubino, Raphael  and  Specia, Lucia  and  Turchi, Marco},\n  title     = {Findings of the 2017 Conference on Machine Translation (WMT17)},\n  booktitle = {Proceedings of the Second Conference on Machine Translation, Volume 2: Shared Task Papers},\n  month     = {September},\n  year      = {2017},\n  address   = {Copenhagen, Denmark},\n  publisher = {Association for Computational Linguistics},\n  pages     = {169--214},\n  url       = {http://www.aclweb.org/anthology/W17-4717}\n}",
        langpairs={
            "cs-en": [
                "test/newstest2017-csen-src.cs.sgm",
                "test/newstest2017-csen-ref.en.sgm",
            ],
            "de-en": [
                "test/newstest2017-deen-src.de.sgm",
                "test/newstest2017-deen-ref.en.sgm",
            ],
            "en-cs": [
                "test/newstest2017-encs-src.en.sgm",
                "test/newstest2017-encs-ref.cs.sgm",
            ],
            "en-de": [
                "test/newstest2017-ende-src.en.sgm",
                "test/newstest2017-ende-ref.de.sgm",
            ],
            "en-fi": [
                "test/newstest2017-enfi-src.en.sgm",
                "test/newstest2017-enfi-ref.fi.sgm",
            ],
            "en-lv": [
                "test/newstest2017-enlv-src.en.sgm",
                "test/newstest2017-enlv-ref.lv.sgm",
            ],
            "en-ru": [
                "test/newstest2017-enru-src.en.sgm",
                "test/newstest2017-enru-ref.ru.sgm",
            ],
            "en-tr": [
                "test/newstest2017-entr-src.en.sgm",
                "test/newstest2017-entr-ref.tr.sgm",
            ],
            "en-zh": [
                "test/newstest2017-enzh-src.en.sgm",
                "test/newstest2017-enzh-ref.zh.sgm",
            ],
            "fi-en": [
                "test/newstest2017-fien-src.fi.sgm",
                "test/newstest2017-fien-ref.en.sgm",
            ],
            "lv-en": [
                "test/newstest2017-lven-src.lv.sgm",
                "test/newstest2017-lven-ref.en.sgm",
            ],
            "ru-en": [
                "test/newstest2017-ruen-src.ru.sgm",
                "test/newstest2017-ruen-ref.en.sgm",
            ],
            "tr-en": [
                "test/newstest2017-tren-src.tr.sgm",
                "test/newstest2017-tren-ref.en.sgm",
            ],
            "zh-en": [
                "test/newstest2017-zhen-src.zh.sgm",
                "test/newstest2017-zhen-ref.en.sgm",
            ],
        },
    ),
    "wmt17/B": FakeSGMLDataset(
        "wmt17/B",
        data=["http://data.statmt.org/wmt17/translation-task/test.tgz"],
        md5=["86a1724c276004aa25455ae2a04cef26"],
        description="Additional reference for EN-FI and FI-EN.",
        langpairs={
            "en-fi": [
                "test/newstestB2017-enfi-src.en.sgm",
                "test/newstestB2017-enfi-ref.fi.sgm",
            ],
        },
    ),
    "wmt17/tworefs": FakeSGMLDataset(
        "wmt17/tworefs",
        data=["http://data.statmt.org/wmt17/translation-task/test.tgz"],
        md5=["86a1724c276004aa25455ae2a04cef26"],
        description="Systems with two references.",
        langpairs={
            "en-fi": [
                "test/newstest2017-enfi-src.en.sgm",
                "test/newstest2017-enfi-ref.fi.sgm",
                "test/newstestB2017-enfi-ref.fi.sgm",
            ],
        },
    ),
    "wmt17/improved": FakeSGMLDataset(
        "wmt17/improved",
        data=["http://data.statmt.org/wmt17/translation-task/test-update-1.tgz"],
        md5=["91dbfd5af99bc6891a637a68e04dfd41"],
        description="Improved zh-en and en-zh translations.",
        langpairs={
            "en-zh": ["newstest2017-enzh-src.en.sgm", "newstest2017-enzh-ref.zh.sgm"],
            "zh-en": ["newstest2017-zhen-src.zh.sgm", "newstest2017-zhen-ref.en.sgm"],
        },
    ),
    "wmt17/dev": FakeSGMLDataset(
        "wmt17/dev",
        data=["http://data.statmt.org/wmt17/translation-task/dev.tgz"],
        md5=["9b1aa63c1cf49dccdd20b962fe313989"],
        description="Development sets released for new languages in 2017.",
        langpairs={
            "en-lv": [
                "dev/newsdev2017-enlv-src.en.sgm",
                "dev/newsdev2017-enlv-ref.lv.sgm",
            ],
            "en-zh": [
                "dev/newsdev2017-enzh-src.en.sgm",
                "dev/newsdev2017-enzh-ref.zh.sgm",
            ],
            "lv-en": [
                "dev/newsdev2017-lven-src.lv.sgm",
                "dev/newsdev2017-lven-ref.en.sgm",
            ],
            "zh-en": [
                "dev/newsdev2017-zhen-src.zh.sgm",
                "dev/newsdev2017-zhen-ref.en.sgm",
            ],
        },
    ),
    "wmt17/ms": WMTAdditionDataset(
        "wmt17/ms",
        data=[
            "https://github.com/MicrosoftTranslator/Translator-HumanParityData/archive/master.zip",
            "http://data.statmt.org/wmt17/translation-task/test-update-1.tgz",
        ],
        md5=["18fdaa7a3c84cf6ef688da1f6a5fa96f", "91dbfd5af99bc6891a637a68e04dfd41"],
        description="Additional Chinese-English references from Microsoft Research.",
        citation="@inproceedings{achieving-human-parity-on-automatic-chinese-to-english-news-translation,\n  author = {Hassan Awadalla, Hany and Aue, Anthony and Chen, Chang and Chowdhary, Vishal and Clark, Jonathan and Federmann, Christian and Huang, Xuedong and Junczys-Dowmunt, Marcin and Lewis, Will and Li, Mu and Liu, Shujie and Liu, Tie-Yan and Luo, Renqian and Menezes, Arul and Qin, Tao and Seide, Frank and Tan, Xu and Tian, Fei and Wu, Lijun and Wu, Shuangzhi and Xia, Yingce and Zhang, Dongdong and Zhang, Zhirui and Zhou, Ming},\n  title = {Achieving Human Parity on Automatic Chinese to English News Translation},\n  booktitle = {},\n  year = {2018},\n  month = {March},\n  abstract = {Machine translation has made rapid advances in recent years. Millions of people are using it today in online translation systems and mobile applications in order to communicate across language barriers. The question naturally arises whether such systems can approach or achieve parity with human translations. In this paper, we first address the problem of how to define and accurately measure human parity in translation. We then describe Microsoft’s machine translation system and measure the quality of its translations on the widely used WMT 2017 news translation task from Chinese to English. We find that our latest neural machine translation system has reached a new state-of-the-art, and that the translation quality is at human parity when compared to professional human translations. We also find that it significantly exceeds the quality of crowd-sourced non-professional translations.},\n  publisher = {},\n  url = {https://www.microsoft.com/en-us/research/publication/achieving-human-parity-on-automatic-chinese-to-english-news-translation/},\n  address = {},\n  pages = {},\n  journal = {},\n  volume = {},\n  chapter = {},\n  isbn = {},\n}",
        langpairs={
            "zh-en": [
                "newstest2017-zhen-src.zh.sgm",
                "newstest2017-zhen-ref.en.sgm",
                "Translator-HumanParityData-master/Translator-HumanParityData/References/Translator-HumanParityData-Reference-HT.txt",
                "Translator-HumanParityData-master/Translator-HumanParityData/References/Translator-HumanParityData-Reference-PE.txt",
            ],
        },
    ),
    "wmt16": FakeSGMLDataset(
        "wmt16",
        data=["http://data.statmt.org/wmt16/translation-task/test.tgz"],
        md5=["3d809cd0c2c86adb2c67034d15c4e446"],
        description="Official evaluation data.",
        citation="@InProceedings{bojar-EtAl:2016:WMT1,\n  author    = {Bojar, Ond\\v{r}ej  and  Chatterjee, Rajen  and  Federmann, Christian  and  Graham, Yvette  and  Haddow, Barry  and  Huck, Matthias  and  Jimeno Yepes, Antonio  and  Koehn, Philipp  and  Logacheva, Varvara  and  Monz, Christof  and  Negri, Matteo  and  Neveol, Aurelie  and  Neves, Mariana  and  Popel, Martin  and  Post, Matt  and  Rubino, Raphael  and  Scarton, Carolina  and  Specia, Lucia  and  Turchi, Marco  and  Verspoor, Karin  and  Zampieri, Marcos},\n  title     = {Findings of the 2016 Conference on Machine Translation},\n  booktitle = {Proceedings of the First Conference on Machine Translation},\n  month     = {August},\n  year      = {2016},\n  address   = {Berlin, Germany},\n  publisher = {Association for Computational Linguistics},\n  pages     = {131--198},\n  url       = {http://www.aclweb.org/anthology/W/W16/W16-2301}\n}",
        langpairs={
            "cs-en": [
                "test/newstest2016-csen-src.cs.sgm",
                "test/newstest2016-csen-ref.en.sgm",
            ],
            "de-en": [
                "test/newstest2016-deen-src.de.sgm",
                "test/newstest2016-deen-ref.en.sgm",
            ],
            "en-cs": [
                "test/newstest2016-encs-src.en.sgm",
                "test/newstest2016-encs-ref.cs.sgm",
            ],
            "en-de": [
                "test/newstest2016-ende-src.en.sgm",
                "test/newstest2016-ende-ref.de.sgm",
            ],
            "en-fi": [
                "test/newstest2016-enfi-src.en.sgm",
                "test/newstest2016-enfi-ref.fi.sgm",
            ],
            "en-ro": [
                "test/newstest2016-enro-src.en.sgm",
                "test/newstest2016-enro-ref.ro.sgm",
            ],
            "en-ru": [
                "test/newstest2016-enru-src.en.sgm",
                "test/newstest2016-enru-ref.ru.sgm",
            ],
            "en-tr": [
                "test/newstest2016-entr-src.en.sgm",
                "test/newstest2016-entr-ref.tr.sgm",
            ],
            "fi-en": [
                "test/newstest2016-fien-src.fi.sgm",
                "test/newstest2016-fien-ref.en.sgm",
            ],
            "ro-en": [
                "test/newstest2016-roen-src.ro.sgm",
                "test/newstest2016-roen-ref.en.sgm",
            ],
            "ru-en": [
                "test/newstest2016-ruen-src.ru.sgm",
                "test/newstest2016-ruen-ref.en.sgm",
            ],
            "tr-en": [
                "test/newstest2016-tren-src.tr.sgm",
                "test/newstest2016-tren-ref.en.sgm",
            ],
        },
    ),
    "wmt16/B": FakeSGMLDataset(
        "wmt16/B",
        data=["http://data.statmt.org/wmt16/translation-task/test.tgz"],
        md5=["3d809cd0c2c86adb2c67034d15c4e446"],
        description="Additional reference for EN-FI.",
        langpairs={
            "en-fi": [
                "test/newstest2016-enfi-src.en.sgm",
                "test/newstestB2016-enfi-ref.fi.sgm",
            ],
        },
    ),
    "wmt16/tworefs": FakeSGMLDataset(
        "wmt16/tworefs",
        data=["http://data.statmt.org/wmt16/translation-task/test.tgz"],
        md5=["3d809cd0c2c86adb2c67034d15c4e446"],
        description="EN-FI with two references.",
        langpairs={
            "en-fi": [
                "test/newstest2016-enfi-src.en.sgm",
                "test/newstest2016-enfi-ref.fi.sgm",
                "test/newstestB2016-enfi-ref.fi.sgm",
            ],
        },
    ),
    "wmt16/dev": FakeSGMLDataset(
        "wmt16/dev",
        data=["http://data.statmt.org/wmt16/translation-task/dev.tgz"],
        md5=["4a3dc2760bb077f4308cce96b06e6af6"],
        description="Development sets released for new languages in 2016.",
        langpairs={
            "en-ro": [
                "dev/newsdev2016-enro-src.en.sgm",
                "dev/newsdev2016-enro-ref.ro.sgm",
            ],
            "en-tr": [
                "dev/newsdev2016-entr-src.en.sgm",
                "dev/newsdev2016-entr-ref.tr.sgm",
            ],
            "ro-en": [
                "dev/newsdev2016-roen-src.ro.sgm",
                "dev/newsdev2016-roen-ref.en.sgm",
            ],
            "tr-en": [
                "dev/newsdev2016-tren-src.tr.sgm",
                "dev/newsdev2016-tren-ref.en.sgm",
            ],
        },
    ),
    "wmt15": FakeSGMLDataset(
        "wmt15",
        data=["http://statmt.org/wmt15/test.tgz"],
        md5=["67e3beca15e69fe3d36de149da0a96df"],
        description="Official evaluation data.",
        citation="@InProceedings{bojar-EtAl:2015:WMT,\n  author    = {Bojar, Ond\\v{r}ej  and  Chatterjee, Rajen  and  Federmann, Christian  and  Haddow, Barry  and  Huck, Matthias  and  Hokamp, Chris  and  Koehn, Philipp  and  Logacheva, Varvara  and  Monz, Christof  and  Negri, Matteo  and  Post, Matt  and  Scarton, Carolina  and  Specia, Lucia  and  Turchi, Marco},\n  title     = {Findings of the 2015 Workshop on Statistical Machine Translation},\n  booktitle = {Proceedings of the Tenth Workshop on Statistical Machine Translation},\n  month     = {September},\n  year      = {2015},\n  address   = {Lisbon, Portugal},\n  publisher = {Association for Computational Linguistics},\n  pages     = {1--46},\n  url       = {http://aclweb.org/anthology/W15-3001}\n}",
        langpairs={
            "en-fr": [
                "test/newsdiscusstest2015-enfr-src.en.sgm",
                "test/newsdiscusstest2015-enfr-ref.fr.sgm",
            ],
            "fr-en": [
                "test/newsdiscusstest2015-fren-src.fr.sgm",
                "test/newsdiscusstest2015-fren-ref.en.sgm",
            ],
            "cs-en": [
                "test/newstest2015-csen-src.cs.sgm",
                "test/newstest2015-csen-ref.en.sgm",
            ],
            "de-en": [
                "test/newstest2015-deen-src.de.sgm",
                "test/newstest2015-deen-ref.en.sgm",
            ],
            "en-cs": [
                "test/newstest2015-encs-src.en.sgm",
                "test/newstest2015-encs-ref.cs.sgm",
            ],
            "en-de": [
                "test/newstest2015-ende-src.en.sgm",
                "test/newstest2015-ende-ref.de.sgm",
            ],
            "en-fi": [
                "test/newstest2015-enfi-src.en.sgm",
                "test/newstest2015-enfi-ref.fi.sgm",
            ],
            "en-ru": [
                "test/newstest2015-enru-src.en.sgm",
                "test/newstest2015-enru-ref.ru.sgm",
            ],
            "fi-en": [
                "test/newstest2015-fien-src.fi.sgm",
                "test/newstest2015-fien-ref.en.sgm",
            ],
            "ru-en": [
                "test/newstest2015-ruen-src.ru.sgm",
                "test/newstest2015-ruen-ref.en.sgm",
            ],
        },
    ),
    "wmt14": FakeSGMLDataset(
        "wmt14",
        data=["http://statmt.org/wmt14/test-filtered.tgz"],
        md5=["84c597844c1542e29c2aff23aaee4310"],
        description="Official evaluation data.",
        citation="@InProceedings{bojar-EtAl:2014:W14-33,\n  author    = {Bojar, Ondrej  and  Buck, Christian  and  Federmann, Christian  and  Haddow, Barry  and  Koehn, Philipp  and  Leveling, Johannes  and  Monz, Christof  and  Pecina, Pavel  and  Post, Matt  and  Saint-Amand, Herve  and  Soricut, Radu  and  Specia, Lucia  and  Tamchyna, Ale\\v{s}},\n  title     = {Findings of the 2014 Workshop on Statistical Machine Translation},\n  booktitle = {Proceedings of the Ninth Workshop on Statistical Machine Translation},\n  month     = {June},\n  year      = {2014},\n  address   = {Baltimore, Maryland, USA},\n  publisher = {Association for Computational Linguistics},\n  pages     = {12--58},\n  url       = {http://www.aclweb.org/anthology/W/W14/W14-3302}\n}",
        langpairs={
            "cs-en": [
                "test/newstest2014-csen-src.cs.sgm",
                "test/newstest2014-csen-ref.en.sgm",
            ],
            "en-cs": [
                "test/newstest2014-csen-src.en.sgm",
                "test/newstest2014-csen-ref.cs.sgm",
            ],
            "de-en": [
                "test/newstest2014-deen-src.de.sgm",
                "test/newstest2014-deen-ref.en.sgm",
            ],
            "en-de": [
                "test/newstest2014-deen-src.en.sgm",
                "test/newstest2014-deen-ref.de.sgm",
            ],
            "en-fr": [
                "test/newstest2014-fren-src.en.sgm",
                "test/newstest2014-fren-ref.fr.sgm",
            ],
            "fr-en": [
                "test/newstest2014-fren-src.fr.sgm",
                "test/newstest2014-fren-ref.en.sgm",
            ],
            "en-hi": [
                "test/newstest2014-hien-src.en.sgm",
                "test/newstest2014-hien-ref.hi.sgm",
            ],
            "hi-en": [
                "test/newstest2014-hien-src.hi.sgm",
                "test/newstest2014-hien-ref.en.sgm",
            ],
            "en-ru": [
                "test/newstest2014-ruen-src.en.sgm",
                "test/newstest2014-ruen-ref.ru.sgm",
            ],
            "ru-en": [
                "test/newstest2014-ruen-src.ru.sgm",
                "test/newstest2014-ruen-ref.en.sgm",
            ],
        },
    ),
    "wmt14/full": FakeSGMLDataset(
        "wmt14/full",
        data=["http://statmt.org/wmt14/test-full.tgz"],
        md5=["a8cd784e006feb32ac6f3d9ec7eb389a"],
        description="Evaluation data released after official evaluation for further research.",
        langpairs={
            "cs-en": [
                "test-full/newstest2014-csen-src.cs.sgm",
                "test-full/newstest2014-csen-ref.en.sgm",
            ],
            "en-cs": [
                "test-full/newstest2014-csen-src.en.sgm",
                "test-full/newstest2014-csen-ref.cs.sgm",
            ],
            "de-en": [
                "test-full/newstest2014-deen-src.de.sgm",
                "test-full/newstest2014-deen-ref.en.sgm",
            ],
            "en-de": [
                "test-full/newstest2014-deen-src.en.sgm",
                "test-full/newstest2014-deen-ref.de.sgm",
            ],
            "en-fr": [
                "test-full/newstest2014-fren-src.en.sgm",
                "test-full/newstest2014-fren-ref.fr.sgm",
            ],
            "fr-en": [
                "test-full/newstest2014-fren-src.fr.sgm",
                "test-full/newstest2014-fren-ref.en.sgm",
            ],
            "en-hi": [
                "test-full/newstest2014-hien-src.en.sgm",
                "test-full/newstest2014-hien-ref.hi.sgm",
            ],
            "hi-en": [
                "test-full/newstest2014-hien-src.hi.sgm",
                "test-full/newstest2014-hien-ref.en.sgm",
            ],
            "en-ru": [
                "test-full/newstest2014-ruen-src.en.sgm",
                "test-full/newstest2014-ruen-ref.ru.sgm",
            ],
            "ru-en": [
                "test-full/newstest2014-ruen-src.ru.sgm",
                "test-full/newstest2014-ruen-ref.en.sgm",
            ],
        },
    ),
    "wmt13": FakeSGMLDataset(
        "wmt13",
        data=["http://statmt.org/wmt13/test.tgz"],
        md5=["48eca5d02f637af44e85186847141f67"],
        description="Official evaluation data.",
        citation="@InProceedings{bojar-EtAl:2013:WMT,\n  author    = {Bojar, Ond\\v{r}ej  and  Buck, Christian  and  Callison-Burch, Chris  and  Federmann, Christian  and  Haddow, Barry  and  Koehn, Philipp  and  Monz, Christof  and  Post, Matt  and  Soricut, Radu  and  Specia, Lucia},\n  title     = {Findings of the 2013 {Workshop on Statistical Machine Translation}},\n  booktitle = {Proceedings of the Eighth Workshop on Statistical Machine Translation},\n  month     = {August},\n  year      = {2013},\n  address   = {Sofia, Bulgaria},\n  publisher = {Association for Computational Linguistics},\n  pages     = {1--44},\n  url       = {http://www.aclweb.org/anthology/W13-2201}\n}",
        langpairs={
            "cs-en": ["test/newstest2013-src.cs.sgm", "test/newstest2013-src.en.sgm"],
            "en-cs": ["test/newstest2013-src.en.sgm", "test/newstest2013-src.cs.sgm"],
            "de-en": ["test/newstest2013-src.de.sgm", "test/newstest2013-src.en.sgm"],
            "en-de": ["test/newstest2013-src.en.sgm", "test/newstest2013-src.de.sgm"],
            "es-en": ["test/newstest2013-src.es.sgm", "test/newstest2013-src.en.sgm"],
            "en-es": ["test/newstest2013-src.en.sgm", "test/newstest2013-src.es.sgm"],
            "fr-en": ["test/newstest2013-src.fr.sgm", "test/newstest2013-src.en.sgm"],
            "en-fr": ["test/newstest2013-src.en.sgm", "test/newstest2013-src.fr.sgm"],
            "ru-en": ["test/newstest2013-src.ru.sgm", "test/newstest2013-src.en.sgm"],
            "en-ru": ["test/newstest2013-src.en.sgm", "test/newstest2013-src.ru.sgm"],
        },
    ),
    "wmt12": FakeSGMLDataset(
        "wmt12",
        data=["http://statmt.org/wmt12/test.tgz"],
        md5=["608232d34ebc4ba2ff70fead45674e47"],
        description="Official evaluation data.",
        citation="@InProceedings{callisonburch-EtAl:2012:WMT,\n  author    = {Callison-Burch, Chris  and  Koehn, Philipp  and  Monz, Christof  and  Post, Matt  and  Soricut, Radu  and  Specia, Lucia},\n  title     = {Findings of the 2012 Workshop on Statistical Machine Translation},\n  booktitle = {Proceedings of the Seventh Workshop on Statistical Machine Translation},\n  month     = {June},\n  year      = {2012},\n  address   = {Montr{'e}al, Canada},\n  publisher = {Association for Computational Linguistics},\n  pages     = {10--51},\n  url       = {http://www.aclweb.org/anthology/W12-3102}\n}",
        langpairs={
            "cs-en": ["test/newstest2012-src.cs.sgm", "test/newstest2012-src.en.sgm"],
            "en-cs": ["test/newstest2012-src.en.sgm", "test/newstest2012-src.cs.sgm"],
            "de-en": ["test/newstest2012-src.de.sgm", "test/newstest2012-src.en.sgm"],
            "en-de": ["test/newstest2012-src.en.sgm", "test/newstest2012-src.de.sgm"],
            "es-en": ["test/newstest2012-src.es.sgm", "test/newstest2012-src.en.sgm"],
            "en-es": ["test/newstest2012-src.en.sgm", "test/newstest2012-src.es.sgm"],
            "fr-en": ["test/newstest2012-src.fr.sgm", "test/newstest2012-src.en.sgm"],
            "en-fr": ["test/newstest2012-src.en.sgm", "test/newstest2012-src.fr.sgm"],
        },
    ),
    "wmt11": FakeSGMLDataset(
        "wmt11",
        data=["http://statmt.org/wmt11/test.tgz"],
        md5=["b0c9680adf32d394aefc2b24e3a5937e"],
        description="Official evaluation data.",
        citation="@InProceedings{callisonburch-EtAl:2011:WMT,\n  author    = {Callison-Burch, Chris  and  Koehn, Philipp  and  Monz, Christof  and  Zaidan, Omar},\n  title     = {Findings of the 2011 Workshop on Statistical Machine Translation},\n  booktitle = {Proceedings of the Sixth Workshop on Statistical Machine Translation},\n  month     = {July},\n  year      = {2011},\n  address   = {Edinburgh, Scotland},\n  publisher = {Association for Computational Linguistics},\n  pages     = {22--64},\n  url       = {http://www.aclweb.org/anthology/W11-2103}\n}",
        langpairs={
            "cs-en": ["newstest2011-src.cs.sgm", "newstest2011-src.en.sgm"],
            "en-cs": ["newstest2011-src.en.sgm", "newstest2011-src.cs.sgm"],
            "de-en": ["newstest2011-src.de.sgm", "newstest2011-src.en.sgm"],
            "en-de": ["newstest2011-src.en.sgm", "newstest2011-src.de.sgm"],
            "fr-en": ["newstest2011-src.fr.sgm", "newstest2011-src.en.sgm"],
            "en-fr": ["newstest2011-src.en.sgm", "newstest2011-src.fr.sgm"],
            "es-en": ["newstest2011-src.es.sgm", "newstest2011-src.en.sgm"],
            "en-es": ["newstest2011-src.en.sgm", "newstest2011-src.es.sgm"],
        },
    ),
    "wmt10": FakeSGMLDataset(
        "wmt10",
        data=["http://statmt.org/wmt10/test.tgz"],
        md5=["491cb885a355da5a23ea66e7b3024d5c"],
        description="Official evaluation data.",
        citation="@InProceedings{callisonburch-EtAl:2010:WMT,\n  author    = {Callison-Burch, Chris  and  Koehn, Philipp  and  Monz, Christof  and  Peterson, Kay  and  Przybocki, Mark  and  Zaidan, Omar},\n  title     = {Findings of the 2010 Joint Workshop on Statistical Machine Translation and Metrics for Machine Translation},\n  booktitle = {Proceedings of the Joint Fifth Workshop on Statistical Machine Translation and MetricsMATR},\n  month     = {July},\n  year      = {2010},\n  address   = {Uppsala, Sweden},\n  publisher = {Association for Computational Linguistics},\n  pages     = {17--53},\n  note      = {Revised August 2010},\n  url       = {http://www.aclweb.org/anthology/W10-1703}\n}",
        langpairs={
            "cs-en": ["test/newstest2010-src.cz.sgm", "test/newstest2010-src.en.sgm"],
            "en-cs": ["test/newstest2010-src.en.sgm", "test/newstest2010-src.cz.sgm"],
            "de-en": ["test/newstest2010-src.de.sgm", "test/newstest2010-src.en.sgm"],
            "en-de": ["test/newstest2010-src.en.sgm", "test/newstest2010-src.de.sgm"],
            "es-en": ["test/newstest2010-src.es.sgm", "test/newstest2010-src.en.sgm"],
            "en-es": ["test/newstest2010-src.en.sgm", "test/newstest2010-src.es.sgm"],
            "fr-en": ["test/newstest2010-src.fr.sgm", "test/newstest2010-src.en.sgm"],
            "en-fr": ["test/newstest2010-src.en.sgm", "test/newstest2010-src.fr.sgm"],
        },
    ),
    "wmt09": FakeSGMLDataset(
        "wmt09",
        data=["http://statmt.org/wmt09/test.tgz"],
        md5=["da227abfbd7b666ec175b742a0d27b37"],
        description="Official evaluation data.",
        citation="@InProceedings{callisonburch-EtAl:2009:WMT-09,\n  author    = {Callison-Burch, Chris  and  Koehn, Philipp  and  Monz, Christof  and  Schroeder, Josh},\n  title     = {Findings of the 2009 {W}orkshop on {S}tatistical {M}achine {T}ranslation},\n  booktitle = {Proceedings of the Fourth Workshop on Statistical Machine Translation},\n  month     = {March},\n  year      = {2009},\n  address   = {Athens, Greece},\n  publisher = {Association for Computational Linguistics},\n  pages     = {1--28},\n  url       = {http://www.aclweb.org/anthology/W/W09/W09-0401}\n}",
        langpairs={
            "cs-en": ["test/newstest2009-src.cz.sgm", "test/newstest2009-src.en.sgm"],
            "en-cs": ["test/newstest2009-src.en.sgm", "test/newstest2009-src.cz.sgm"],
            "de-en": ["test/newstest2009-src.de.sgm", "test/newstest2009-src.en.sgm"],
            "en-de": ["test/newstest2009-src.en.sgm", "test/newstest2009-src.de.sgm"],
            "es-en": ["test/newstest2009-src.es.sgm", "test/newstest2009-src.en.sgm"],
            "en-es": ["test/newstest2009-src.en.sgm", "test/newstest2009-src.es.sgm"],
            "fr-en": ["test/newstest2009-src.fr.sgm", "test/newstest2009-src.en.sgm"],
            "en-fr": ["test/newstest2009-src.en.sgm", "test/newstest2009-src.fr.sgm"],
            "hu-en": ["test/newstest2009-src.hu.sgm", "test/newstest2009-src.en.sgm"],
            "en-hu": ["test/newstest2009-src.en.sgm", "test/newstest2009-src.hu.sgm"],
            "it-en": ["test/newstest2009-src.it.sgm", "test/newstest2009-src.en.sgm"],
            "en-it": ["test/newstest2009-src.en.sgm", "test/newstest2009-src.it.sgm"],
        },
    ),
    "wmt08": FakeSGMLDataset(
        "wmt08",
        data=["http://statmt.org/wmt08/test.tgz"],
        md5=["0582e4e894a3342044059c894e1aea3d"],
        description="Official evaluation data.",
        citation="@InProceedings{callisonburch-EtAl:2008:WMT,\n  author    = {Callison-Burch, Chris  and  Fordyce, Cameron  and  Koehn, Philipp  and  Monz, Christof  and  Schroeder, Josh},\n  title     = {Further Meta-Evaluation of Machine Translation},\n  booktitle = {Proceedings of the Third Workshop on Statistical Machine Translation},\n  month     = {June},\n  year      = {2008},\n  address   = {Columbus, Ohio},\n  publisher = {Association for Computational Linguistics},\n  pages     = {70--106},\n  url       = {http://www.aclweb.org/anthology/W/W08/W08-0309}\n}",
        langpairs={
            "cs-en": ["test/newstest2008-src.cz.sgm", "test/newstest2008-src.en.sgm"],
            "en-cs": ["test/newstest2008-src.en.sgm", "test/newstest2008-src.cz.sgm"],
            "de-en": ["test/newstest2008-src.de.sgm", "test/newstest2008-src.en.sgm"],
            "en-de": ["test/newstest2008-src.en.sgm", "test/newstest2008-src.de.sgm"],
            "es-en": ["test/newstest2008-src.es.sgm", "test/newstest2008-src.en.sgm"],
            "en-es": ["test/newstest2008-src.en.sgm", "test/newstest2008-src.es.sgm"],
            "fr-en": ["test/newstest2008-src.fr.sgm", "test/newstest2008-src.en.sgm"],
            "en-fr": ["test/newstest2008-src.en.sgm", "test/newstest2008-src.fr.sgm"],
            "hu-en": ["test/newstest2008-src.hu.sgm", "test/newstest2008-src.en.sgm"],
            "en-hu": ["test/newstest2008-src.en.sgm", "test/newstest2008-src.hu.sgm"],
        },
    ),
    "wmt08/nc": FakeSGMLDataset(
        "wmt08/nc",
        data=["http://statmt.org/wmt08/test.tgz"],
        md5=["0582e4e894a3342044059c894e1aea3d"],
        description="Official evaluation data (news commentary).",
        langpairs={
            "cs-en": ["test/nc-test2008-src.cz.sgm", "test/nc-test2008-src.en.sgm"],
            "en-cs": ["test/nc-test2008-src.en.sgm", "test/nc-test2008-src.cz.sgm"],
        },
    ),
    "wmt08/europarl": FakeSGMLDataset(
        "wmt08/europarl",
        data=["http://statmt.org/wmt08/test.tgz"],
        md5=["0582e4e894a3342044059c894e1aea3d"],
        description="Official evaluation data (Europarl).",
        langpairs={
            "de-en": ["test/test2008-src.de.sgm", "test/test2008-src.en.sgm"],
            "en-de": ["test/test2008-src.en.sgm", "test/test2008-src.de.sgm"],
            "es-en": ["test/test2008-src.es.sgm", "test/test2008-src.en.sgm"],
            "en-es": ["test/test2008-src.en.sgm", "test/test2008-src.es.sgm"],
            "fr-en": ["test/test2008-src.fr.sgm", "test/test2008-src.en.sgm"],
            "en-fr": ["test/test2008-src.en.sgm", "test/test2008-src.fr.sgm"],
        },
    ),
}
