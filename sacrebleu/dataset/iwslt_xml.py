from .base import Dataset


class IWSLTXMLDataset(Dataset):
    """IWSLT dataset format. Can be parsed with the lxml parser."""

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


IWSLT_XML_DATASETS = {
    "iwslt17": IWSLTXMLDataset(
        "iwslt17",
        data=[
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/en/fr/en-fr.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/fr/en/fr-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/en/de/en-de.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/de/en/de-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/en/ar/en-ar.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/ar/en/ar-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/en/ja/en-ja.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/ja/en/ja-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/en/ko/en-ko.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/ko/en/ko-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/en/zh/en-zh.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/zh/en/zh-en.tgz",
        ],
        md5=[
            "1849bcc3b006dc0642a8843b11aa7192",
            "79bf7a2ef02d226875f55fb076e7e473",
            "b68e7097b179491f6c466ef41ad72b9b",
            "e3f5b2a075a2da1a395c8b60bf1e9be1",
            "ecdc6bc4ab4c8984e919444f3c05183a",
            "4b5141d14b98706c081371e2f8afe0ca",
            "d957ee79de1f33c89077d37c5a2c5b06",
            "c213e8bb918ebf843543fe9fd2e33db2",
            "59f6a81c707378176e9ad8bb8d811f5f",
            "7e580af973bb389ec1d1378a1850742f",
            "975a858783a0ebec8c57d83ddd5bd381",
            "cc51d9b7fe1ff2af858c6a0dd80b8815",
        ],
        description="Official evaluation data for IWSLT.",
        citation="@InProceedings{iwslt2017,\n  author    = {Cettolo, Mauro and Federico, Marcello and Bentivogli, Luisa and Niehues, Jan and Stüker, Sebastian and Sudoh, Katsuitho and Yoshino, Koichiro and Federmann, Christian},\n  title     = {Overview of the IWSLT 2017 Evaluation Campaign},\n  booktitle = {14th International Workshop on Spoken Language Translation},\n  month     = {December},\n  year      = {2017},\n  address   = {Tokyo, Japan},\n  pages     = {2--14},\n  url       = {http://workshop2017.iwslt.org/downloads/iwslt2017_proceeding_v2.pdf}\n}",
        langpairs={
            "en-fr": [
                "en-fr/IWSLT17.TED.tst2017.en-fr.en.xml",
                "fr-en/IWSLT17.TED.tst2017.fr-en.fr.xml",
            ],
            "fr-en": [
                "fr-en/IWSLT17.TED.tst2017.fr-en.fr.xml",
                "en-fr/IWSLT17.TED.tst2017.en-fr.en.xml",
            ],
            "en-de": [
                "en-de/IWSLT17.TED.tst2017.en-de.en.xml",
                "de-en/IWSLT17.TED.tst2017.de-en.de.xml",
            ],
            "de-en": [
                "de-en/IWSLT17.TED.tst2017.de-en.de.xml",
                "en-de/IWSLT17.TED.tst2017.en-de.en.xml",
            ],
            "en-zh": [
                "en-zh/IWSLT17.TED.tst2017.en-zh.en.xml",
                "zh-en/IWSLT17.TED.tst2017.zh-en.zh.xml",
            ],
            "zh-en": [
                "zh-en/IWSLT17.TED.tst2017.zh-en.zh.xml",
                "en-zh/IWSLT17.TED.tst2017.en-zh.en.xml",
            ],
            "en-ar": [
                "en-ar/IWSLT17.TED.tst2017.en-ar.en.xml",
                "ar-en/IWSLT17.TED.tst2017.ar-en.ar.xml",
            ],
            "ar-en": [
                "ar-en/IWSLT17.TED.tst2017.ar-en.ar.xml",
                "en-ar/IWSLT17.TED.tst2017.en-ar.en.xml",
            ],
            "en-ja": [
                "en-ja/IWSLT17.TED.tst2017.en-ja.en.xml",
                "ja-en/IWSLT17.TED.tst2017.ja-en.ja.xml",
            ],
            "ja-en": [
                "ja-en/IWSLT17.TED.tst2017.ja-en.ja.xml",
                "en-ja/IWSLT17.TED.tst2017.en-ja.en.xml",
            ],
            "en-ko": [
                "en-ko/IWSLT17.TED.tst2017.en-ko.en.xml",
                "ko-en/IWSLT17.TED.tst2017.ko-en.ko.xml",
            ],
            "ko-en": [
                "ko-en/IWSLT17.TED.tst2017.ko-en.ko.xml",
                "en-ko/IWSLT17.TED.tst2017.en-ko.en.xml",
            ],
        },
    ),
    "iwslt17/tst2016": IWSLTXMLDataset(
        "iwslt17/tst2016",
        data=[
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/en/fr/en-fr.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/fr/en/fr-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/en/de/en-de.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/de/en/de-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/en/zh/en-zh.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/zh/en/zh-en.tgz",
        ],
        md5=[
            "1849bcc3b006dc0642a8843b11aa7192",
            "79bf7a2ef02d226875f55fb076e7e473",
            "b68e7097b179491f6c466ef41ad72b9b",
            "e3f5b2a075a2da1a395c8b60bf1e9be1",
            "975a858783a0ebec8c57d83ddd5bd381",
            "cc51d9b7fe1ff2af858c6a0dd80b8815",
        ],
        description="Development data for IWSLT 2017.",
        langpairs={
            "en-fr": [
                "en-fr/IWSLT17.TED.tst2016.en-fr.en.xml",
                "fr-en/IWSLT17.TED.tst2016.fr-en.fr.xml",
            ],
            "fr-en": [
                "fr-en/IWSLT17.TED.tst2016.fr-en.fr.xml",
                "en-fr/IWSLT17.TED.tst2016.en-fr.en.xml",
            ],
            "en-de": [
                "en-de/IWSLT17.TED.tst2016.en-de.en.xml",
                "de-en/IWSLT17.TED.tst2016.de-en.de.xml",
            ],
            "de-en": [
                "de-en/IWSLT17.TED.tst2016.de-en.de.xml",
                "en-de/IWSLT17.TED.tst2016.en-de.en.xml",
            ],
            "en-zh": [
                "en-zh/IWSLT17.TED.tst2016.en-zh.en.xml",
                "zh-en/IWSLT17.TED.tst2016.zh-en.zh.xml",
            ],
            "zh-en": [
                "zh-en/IWSLT17.TED.tst2016.zh-en.zh.xml",
                "en-zh/IWSLT17.TED.tst2016.en-zh.en.xml",
            ],
        },
    ),
    "iwslt17/tst2015": IWSLTXMLDataset(
        "iwslt17/tst2015",
        data=[
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/de/en-de.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/de/en/de-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/fr/en-fr.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/fr/en/fr-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/zh/en-zh.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/zh/en/zh-en.tgz",
        ],
        md5=[
            "d8a32cfc002a4f12b17429cfa78050e6",
            "ca2b94d694150d4d6c5dc64c200fa589",
            "3cf07ebe305312b12f7f1a4d5f8f8377",
            "19927da9de0f40348cad9c0fc61642ac",
            "575b788dad6c5b9c5cee636f9ac1094a",
            "1c0ae40171d52593df8a6963d3828116",
        ],
        description="Development data for IWSLT 2017.",
        langpairs={
            "en-fr": [
                "en-fr/IWSLT17.TED.tst2015.en-fr.en.xml",
                "fr-en/IWSLT17.TED.tst2015.fr-en.fr.xml",
            ],
            "fr-en": [
                "fr-en/IWSLT17.TED.tst2015.fr-en.fr.xml",
                "en-fr/IWSLT17.TED.tst2015.en-fr.en.xml",
            ],
            "en-de": [
                "en-de/IWSLT17.TED.tst2015.en-de.en.xml",
                "de-en/IWSLT17.TED.tst2015.de-en.de.xml",
            ],
            "de-en": [
                "de-en/IWSLT17.TED.tst2015.de-en.de.xml",
                "en-de/IWSLT17.TED.tst2015.en-de.en.xml",
            ],
            "en-zh": [
                "en-zh/IWSLT17.TED.tst2015.en-zh.en.xml",
                "zh-en/IWSLT17.TED.tst2015.zh-en.zh.xml",
            ],
            "zh-en": [
                "zh-en/IWSLT17.TED.tst2015.zh-en.zh.xml",
                "en-zh/IWSLT17.TED.tst2015.en-zh.en.xml",
            ],
        },
    ),
    "iwslt17/tst2014": IWSLTXMLDataset(
        "iwslt17/tst2014",
        data=[
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/de/en-de.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/de/en/de-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/fr/en-fr.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/fr/en/fr-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/zh/en-zh.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/zh/en/zh-en.tgz",
        ],
        md5=[
            "d8a32cfc002a4f12b17429cfa78050e6",
            "ca2b94d694150d4d6c5dc64c200fa589",
            "3cf07ebe305312b12f7f1a4d5f8f8377",
            "19927da9de0f40348cad9c0fc61642ac",
            "575b788dad6c5b9c5cee636f9ac1094a",
            "1c0ae40171d52593df8a6963d3828116",
        ],
        description="Development data for IWSLT 2017.",
        langpairs={
            "en-fr": [
                "en-fr/IWSLT17.TED.tst2014.en-fr.en.xml",
                "fr-en/IWSLT17.TED.tst2014.fr-en.fr.xml",
            ],
            "fr-en": [
                "fr-en/IWSLT17.TED.tst2014.fr-en.fr.xml",
                "en-fr/IWSLT17.TED.tst2014.en-fr.en.xml",
            ],
            "en-de": [
                "en-de/IWSLT17.TED.tst2014.en-de.en.xml",
                "de-en/IWSLT17.TED.tst2014.de-en.de.xml",
            ],
            "de-en": [
                "de-en/IWSLT17.TED.tst2014.de-en.de.xml",
                "en-de/IWSLT17.TED.tst2014.en-de.en.xml",
            ],
            "en-zh": [
                "en-zh/IWSLT17.TED.tst2014.en-zh.en.xml",
                "zh-en/IWSLT17.TED.tst2014.zh-en.zh.xml",
            ],
            "zh-en": [
                "zh-en/IWSLT17.TED.tst2014.zh-en.zh.xml",
                "en-zh/IWSLT17.TED.tst2014.en-zh.en.xml",
            ],
        },
    ),
    "iwslt17/tst2013": IWSLTXMLDataset(
        "iwslt17/tst2013",
        data=[
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/de/en-de.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/de/en/de-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/fr/en-fr.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/fr/en/fr-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/zh/en-zh.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/zh/en/zh-en.tgz",
        ],
        md5=[
            "d8a32cfc002a4f12b17429cfa78050e6",
            "ca2b94d694150d4d6c5dc64c200fa589",
            "3cf07ebe305312b12f7f1a4d5f8f8377",
            "19927da9de0f40348cad9c0fc61642ac",
            "575b788dad6c5b9c5cee636f9ac1094a",
            "1c0ae40171d52593df8a6963d3828116",
        ],
        description="Development data for IWSLT 2017.",
        langpairs={
            "en-fr": [
                "en-fr/IWSLT17.TED.tst2013.en-fr.en.xml",
                "fr-en/IWSLT17.TED.tst2013.fr-en.fr.xml",
            ],
            "fr-en": [
                "fr-en/IWSLT17.TED.tst2013.fr-en.fr.xml",
                "en-fr/IWSLT17.TED.tst2013.en-fr.en.xml",
            ],
            "en-de": [
                "en-de/IWSLT17.TED.tst2013.en-de.en.xml",
                "de-en/IWSLT17.TED.tst2013.de-en.de.xml",
            ],
            "de-en": [
                "de-en/IWSLT17.TED.tst2013.de-en.de.xml",
                "en-de/IWSLT17.TED.tst2013.en-de.en.xml",
            ],
            "en-zh": [
                "en-zh/IWSLT17.TED.tst2013.en-zh.en.xml",
                "zh-en/IWSLT17.TED.tst2013.zh-en.zh.xml",
            ],
            "zh-en": [
                "zh-en/IWSLT17.TED.tst2013.zh-en.zh.xml",
                "en-zh/IWSLT17.TED.tst2013.en-zh.en.xml",
            ],
        },
    ),
    "iwslt17/tst2012": IWSLTXMLDataset(
        "iwslt17/tst2012",
        data=[
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/de/en-de.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/de/en/de-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/fr/en-fr.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/fr/en/fr-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/zh/en-zh.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/zh/en/zh-en.tgz",
        ],
        md5=[
            "d8a32cfc002a4f12b17429cfa78050e6",
            "ca2b94d694150d4d6c5dc64c200fa589",
            "3cf07ebe305312b12f7f1a4d5f8f8377",
            "19927da9de0f40348cad9c0fc61642ac",
            "575b788dad6c5b9c5cee636f9ac1094a",
            "1c0ae40171d52593df8a6963d3828116",
        ],
        description="Development data for IWSLT 2017.",
        langpairs={
            "en-fr": [
                "en-fr/IWSLT17.TED.tst2012.en-fr.en.xml",
                "fr-en/IWSLT17.TED.tst2012.fr-en.fr.xml",
            ],
            "fr-en": [
                "fr-en/IWSLT17.TED.tst2012.fr-en.fr.xml",
                "en-fr/IWSLT17.TED.tst2012.en-fr.en.xml",
            ],
            "en-de": [
                "en-de/IWSLT17.TED.tst2012.en-de.en.xml",
                "de-en/IWSLT17.TED.tst2012.de-en.de.xml",
            ],
            "de-en": [
                "de-en/IWSLT17.TED.tst2012.de-en.de.xml",
                "en-de/IWSLT17.TED.tst2012.en-de.en.xml",
            ],
            "en-zh": [
                "en-zh/IWSLT17.TED.tst2012.en-zh.en.xml",
                "zh-en/IWSLT17.TED.tst2012.zh-en.zh.xml",
            ],
            "zh-en": [
                "zh-en/IWSLT17.TED.tst2012.zh-en.zh.xml",
                "en-zh/IWSLT17.TED.tst2012.en-zh.en.xml",
            ],
        },
    ),
    "iwslt17/tst2011": IWSLTXMLDataset(
        "iwslt17/tst2011",
        data=[
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/de/en-de.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/de/en/de-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/fr/en-fr.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/fr/en/fr-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/zh/en-zh.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/zh/en/zh-en.tgz",
        ],
        md5=[
            "d8a32cfc002a4f12b17429cfa78050e6",
            "ca2b94d694150d4d6c5dc64c200fa589",
            "3cf07ebe305312b12f7f1a4d5f8f8377",
            "19927da9de0f40348cad9c0fc61642ac",
            "575b788dad6c5b9c5cee636f9ac1094a",
            "1c0ae40171d52593df8a6963d3828116",
        ],
        description="Development data for IWSLT 2017.",
        langpairs={
            "en-fr": [
                "en-fr/IWSLT17.TED.tst2011.en-fr.en.xml",
                "fr-en/IWSLT17.TED.tst2011.fr-en.fr.xml",
            ],
            "fr-en": [
                "fr-en/IWSLT17.TED.tst2011.fr-en.fr.xml",
                "en-fr/IWSLT17.TED.tst2011.en-fr.en.xml",
            ],
            "en-de": [
                "en-de/IWSLT17.TED.tst2011.en-de.en.xml",
                "de-en/IWSLT17.TED.tst2011.de-en.de.xml",
            ],
            "de-en": [
                "de-en/IWSLT17.TED.tst2011.de-en.de.xml",
                "en-de/IWSLT17.TED.tst2011.en-de.en.xml",
            ],
            "en-zh": [
                "en-zh/IWSLT17.TED.tst2011.en-zh.en.xml",
                "zh-en/IWSLT17.TED.tst2011.zh-en.zh.xml",
            ],
            "zh-en": [
                "zh-en/IWSLT17.TED.tst2011.zh-en.zh.xml",
                "en-zh/IWSLT17.TED.tst2011.en-zh.en.xml",
            ],
        },
    ),
    "iwslt17/tst2010": IWSLTXMLDataset(
        "iwslt17/tst2010",
        data=[
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/de/en-de.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/de/en/de-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/fr/en-fr.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/fr/en/fr-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/zh/en-zh.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/zh/en/zh-en.tgz",
        ],
        md5=[
            "d8a32cfc002a4f12b17429cfa78050e6",
            "ca2b94d694150d4d6c5dc64c200fa589",
            "3cf07ebe305312b12f7f1a4d5f8f8377",
            "19927da9de0f40348cad9c0fc61642ac",
            "575b788dad6c5b9c5cee636f9ac1094a",
            "1c0ae40171d52593df8a6963d3828116",
        ],
        description="Development data for IWSLT 2017.",
        langpairs={
            "en-fr": [
                "en-fr/IWSLT17.TED.tst2010.en-fr.en.xml",
                "fr-en/IWSLT17.TED.tst2010.fr-en.fr.xml",
            ],
            "fr-en": [
                "fr-en/IWSLT17.TED.tst2010.fr-en.fr.xml",
                "en-fr/IWSLT17.TED.tst2010.en-fr.en.xml",
            ],
            "en-de": [
                "en-de/IWSLT17.TED.tst2010.en-de.en.xml",
                "de-en/IWSLT17.TED.tst2010.de-en.de.xml",
            ],
            "de-en": [
                "de-en/IWSLT17.TED.tst2010.de-en.de.xml",
                "en-de/IWSLT17.TED.tst2010.en-de.en.xml",
            ],
            "en-zh": [
                "en-zh/IWSLT17.TED.tst2010.en-zh.en.xml",
                "zh-en/IWSLT17.TED.tst2010.zh-en.zh.xml",
            ],
            "zh-en": [
                "zh-en/IWSLT17.TED.tst2010.zh-en.zh.xml",
                "en-zh/IWSLT17.TED.tst2010.en-zh.en.xml",
            ],
        },
    ),
    "iwslt17/dev2010": IWSLTXMLDataset(
        "iwslt17/dev2010",
        data=[
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/de/en-de.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/de/en/de-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/fr/en-fr.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/fr/en/fr-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/zh/en-zh.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/zh/en/zh-en.tgz",
        ],
        md5=[
            "d8a32cfc002a4f12b17429cfa78050e6",
            "ca2b94d694150d4d6c5dc64c200fa589",
            "3cf07ebe305312b12f7f1a4d5f8f8377",
            "19927da9de0f40348cad9c0fc61642ac",
            "575b788dad6c5b9c5cee636f9ac1094a",
            "1c0ae40171d52593df8a6963d3828116",
        ],
        description="Development data for IWSLT 2017.",
        langpairs={
            "en-fr": [
                "en-fr/IWSLT17.TED.dev2010.en-fr.en.xml",
                "fr-en/IWSLT17.TED.dev2010.fr-en.fr.xml",
            ],
            "fr-en": [
                "fr-en/IWSLT17.TED.dev2010.fr-en.fr.xml",
                "en-fr/IWSLT17.TED.dev2010.en-fr.en.xml",
            ],
            "en-de": [
                "en-de/IWSLT17.TED.dev2010.en-de.en.xml",
                "de-en/IWSLT17.TED.dev2010.de-en.de.xml",
            ],
            "de-en": [
                "de-en/IWSLT17.TED.dev2010.de-en.de.xml",
                "en-de/IWSLT17.TED.dev2010.en-de.en.xml",
            ],
            "en-zh": [
                "en-zh/IWSLT17.TED.dev2010.en-zh.en.xml",
                "zh-en/IWSLT17.TED.dev2010.zh-en.zh.xml",
            ],
            "zh-en": [
                "zh-en/IWSLT17.TED.dev2010.zh-en.zh.xml",
                "en-zh/IWSLT17.TED.dev2010.en-zh.en.xml",
            ],
        },
    ),
}
