from __future__ import annotations

import json
import os
from collections import defaultdict

from ..utils import smart_open
from .base import Dataset


class WMT25Dataset(Dataset):
    """
    Dataset for WMT25 general machine translation task (wmt25-genmt.jsonl).
    """

    NO_REF_LANGPAIRS = {
        "en-bn_BD",
        "en-de_DE",
        "en-el_GR",
        "en-fa_IR",
        "en-hi_IN",
        "en-id_ID",
        "en-it_IT",
        "en-kn_IN",
        "en-lt_LT",
        "en-mr_IN",
        "en-ro_RO",
        "en-sr_Latn_RS",
        "en-sv_SE",
        "en-th_TH",
        "en-tr_TR",
        "en-vi_VN",
    }

    def fieldnames(self, langpair) -> list[str]:
        if langpair in self.NO_REF_LANGPAIRS:
            return ["src"]
        if langpair == "en-cs_CZ":
            return ["src", "ref", "refB"]
        return ["src", "ref"]

    def process_to_text(self, langpair=None) -> None:
        """Processes raw jsonl file to plain text files for all language pairs."""
        self.maybe_download()
        self._get_langpair_metadata(langpair)

        raw_file = os.path.join(self._rawdir, self._get_tarball_filename(self.data[0]))

        data_by_lp: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: {"src": [], "ref": [], "refB": []}
        )

        with smart_open(raw_file) as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                src_l = item.get("src_lang")
                tgt_l = item.get("tgt_lang")
                lp = f"{src_l}-{tgt_l}"
                if lp not in self.langpairs:
                    continue

                src_text = self._clean(item.get("src_text", ""))
                data_by_lp[lp]["src"].append(src_text)

                refs = item.get("refs", {})
                if "refA" in refs and "ref" in refs["refA"]:
                    refA = self._clean(refs["refA"]["ref"])
                    data_by_lp[lp]["ref"].append(refA)

                    if lp == "en-cs_CZ":
                        if "refB" in refs and "ref" in refs["refB"]:
                            refB = self._clean(refs["refB"]["ref"])
                        else:
                            refB = refA
                        data_by_lp[lp]["refB"].append(refB)

        os.makedirs(self._outdir, exist_ok=True)
        for lp in self.langpairs:
            if lp not in data_by_lp:
                continue
            fields = self.fieldnames(lp)
            for field in fields:
                out_file = self._get_txt_file_path(lp, field)
                with smart_open(out_file, "wt") as fout:
                    for text_line in data_by_lp[lp][field]:
                        print(text_line, file=fout)
