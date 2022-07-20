import os

import lxml.etree as ET

from ..utils import smart_open
from .base import Dataset


class WMTXMLDataset(Dataset):
    """
    The 2021+ WMT dataset format. Everything is contained in a single file.
    Can be parsed with the lxml parser.
    """

    @staticmethod
    def _unwrap_wmt21_or_later(raw_file):
        """
        Unwraps the XML file from wmt21 or later.
        This script is adapted from https://github.com/wmt-conference/wmt-format-tools

        :param raw_file: The raw xml file to unwrap.
        :return: Dictionary which contains the following fields:
            - `src`: The source sentences.
            - `docid`: ID indicating which document the sentences belong to.
            - `origlang`: The original language of the document.
            - `ref`: The reference sentences unknown translator.
            - `ref:A`: Reference from translator A.
            - `ref:B`: Reference from translator B.
            - `ref:C`: Reference from translator C.
            - `ref:D`: Reference from translator D.
        """
        tree = ET.parse(raw_file)
        # Find and check  the documents (src, ref, hyp)
        src_langs, ref_langs, translators = set(), set(), set()
        for src_doc in tree.getroot().findall(".//src"):
            src_langs.add(src_doc.get("lang"))

        for ref_doc in tree.getroot().findall(".//ref"):
            ref_langs.add(ref_doc.get("lang"))
            translator = ref_doc.get("translator")
            translators.add(translator)

        assert (
            len(src_langs) == 1
        ), f"Multiple source languages found in the file: {raw_file}"
        assert (
            len(ref_langs) == 1
        ), f"Multiple reference languages found in the file: {raw_file}"
        src = []
        docids = []
        orig_langs = []

        def get_field_by_translator(translator):
            if not translator:
                return "ref"
            else:
                return f"ref:{translator}"

        refs = {get_field_by_translator(translator): [] for translator in translators}

        src_sent_count, doc_count = 0, 0
        for doc in tree.getroot().findall(".//doc"):
            docid = doc.attrib["id"]
            origlang = doc.attrib["origlang"]

            # Skip the testsuite
            if "testsuite" in doc.attrib:
                continue

            doc_count += 1
            src_sents = {
                int(seg.get("id")): seg.text for seg in doc.findall(".//src//seg")
            }

            def get_sents(doc):
                return {
                    int(seg.get("id")): seg.text if seg.text else ""
                    for seg in doc.findall(f".//seg")
                }

            ref_docs = doc.findall(".//ref")

            trans_to_ref = {
                ref_doc.get("translator"): get_sents(ref_doc) for ref_doc in ref_docs
            }

            for seg_id in sorted(src_sents.keys()):
                # no ref translation is avaliable for this segment
                if not any([value.get(seg_id, "") for value in trans_to_ref.values()]):
                    continue
                for translator in translators:
                    refs[get_field_by_translator(translator)].append(
                        trans_to_ref.get(translator, {translator: {}}).get(seg_id, "")
                    )
                src.append(src_sents[seg_id])
                docids.append(docid)
                orig_langs.append(origlang)
                src_sent_count += 1

        # For backward compatibility, if "ref" is not in the fields,
        # add reference seneteces from the first translator as "ref" field
        if "ref" not in refs:
            refs["ref"] = refs[min(refs.keys())]

        return {"src": src, **refs, "docid": docids, "origlang": orig_langs,}

    def process_to_text(self, langpair=None):
        """Processes raw files to plain text files.

        :param langpair: The language pair to process. e.g. "en-de". If None, all files will be processed.
        """
        # ensure that the dataset is downloaded
        self.maybe_download()
        langpairs = self._get_langpair_metadata(langpair)

        for langpair, files in langpairs.items():
            rawfile = os.path.join(
                self._rawdir, files[0]
            )  # all source and reference data in one file for wmt21 and later

            with smart_open(rawfile) as fin:
                fields = self._unwrap_wmt21_or_later(fin)

            for fieldname in fields:
                textfile = self._get_txt_file_path(langpair, fieldname)

                # skip if the file already exists
                if os.path.exists(textfile) and os.path.getsize(textfile) > 0:
                    continue

                with smart_open(textfile, "w") as fout:
                    for line in fields[fieldname]:
                        print(self._clean(line), file=fout)

    def fieldnames(self, langpair):
        """
        Return a list of all the field names. For most source, this is just
        the source and the reference. For others, it might include the document
        ID for each line, or the original language (origLang).

        get_files() should return the same number of items as this.

        :param langpair: The language pair (e.g., "de-en")
        :return: a list of field names
        """
        self.maybe_download()
        meta = self._get_langpair_metadata(langpair)[langpair]
        rawfile = os.path.join(self._rawdir, meta[0])

        with smart_open(rawfile) as fin:
            fields = self._unwrap_wmt21_or_later(fin)

        return list(fields.keys())


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
            "en-ha": ["dev/xml/newsdev2021.en-ha.xml"],
            "ha-en": ["dev/xml/newsdev2021.ha-en.xml"],
            "en-is": ["dev/xml/newsdev2021.en-is.xml"],
            "is-en": ["dev/xml/newsdev2021.is-en.xml"],
        },
    ),
}
