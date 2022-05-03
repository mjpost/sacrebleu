from .base import Dataset


class WMTXMLDataset(Dataset):
    """
    The 2021+ WMT dataset format. Everything is contained in a single file.
    Can be parsed with the lxml parser.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


WMT_XML_DATASETS = {
    "wmt21": WMTXMLDataset(
        "wmt21",
        data=["http://data.statmt.org/wmt21/translation-task/test.tgz"],
        description="Official evaluation data for WMT21. If multiple references are available, the first one is used.",
        md5=["32e7ab995bc318414375d60f0269af92"],
        langpairs={
            "de-fr": ["test/newstest2021.de-fr.xml"],
            "en-de": ["test/newstest2021.en-de.xml"],
            "en-ha": ["test/newstest2021.en-ha.xml"],
            "en-is": ["test/newstest2021.en-is.xml"],
            "en-ja": ["test/newstest2021.en-ja.xml"],
            "fr-de": ["test/newstest2021.fr-de.xml"],
            "ha-en": ["test/newstest2021.ha-en.xml"],
            "is-en": ["test/newstest2021.is-en.xml"],
            "ja-en": ["test/newstest2021.ja-en.xml"],
            "zh-en": ["test/newstest2021.zh-en.xml"],
            "en-zh": ["test/newstest2021.en-zh.xml"],
            "cs-en": ["test/newstest2021.cs-en.xml"],
            "de-en": ["test/newstest2021.de-en.xml"],
            "en-cs": ["test/newstest2021.en-cs.xml"],
            "en-ru": ["test/newstest2021.en-ru.xml"],
            "ru-en": ["test/newstest2021.ru-en.xml"],
            "bn-hi": ["test/florestest2021.bn-hi.xml"],
            "hi-bn": ["test/florestest2021.hi-bn.xml"],
            "xh-zu": ["test/florestest2021.xh-zu.xml"],
            "zu-xh": ["test/florestest2021.zu-xh.xml"],
        },
    ),
    "wmt21/dev": WMTXMLDataset(
        "wmt21/dev",
        data=["http://data.statmt.org/wmt21/translation-task/dev.tgz"],
        description="Development data for WMT21，if multiple references are available, the first one is used.",
        md5=["165da59ac8dfb5b7cafd7e90b1cac672"],
        langpairs={
            "en-ha": ["dev/xml/newsdev2021.en-ha.xml", "dev/xml/newsdev2021.en-ha.xml"],
            "ha-en": ["dev/xml/newsdev2021.ha-en.xml", "dev/xml/newsdev2021.ha-en.xml"],
            "en-is": ["dev/xml/newsdev2021.en-is.xml", "dev/xml/newsdev2021.en-is.xml"],
            "is-en": ["dev/xml/newsdev2021.is-en.xml", "dev/xml/newsdev2021.is-en.xml"],
        },
    ),
}
