from __future__ import annotations

import json
import os

from ..utils import smart_open
from .base import Dataset


class WMT26Dataset(Dataset):
    """
    Dataset for WMT26 evaluation campaigns (campaign_sources.zip).
    """

    REF_PREFERENCE = [
        "Human (from scratch)",
        "Human (postediting)",
        "PseudoRef",
    ]

    def fieldnames(self, langpair) -> list[str]:
        if langpair == "ces_Latn-ukr_Cyrl":
            return ["src"]
        return ["src", "ref"]

    def process_to_text(self, langpair=None) -> None:
        """Processes raw JSON files in rawdir to plain text files."""
        self.maybe_download()
        self._get_langpair_metadata(langpair)

        os.makedirs(self._outdir, exist_ok=True)

        for lp in self.langpairs:
            json_filename = lp.replace("-", "---") + ".json"
            json_path = os.path.join(self._rawdir, json_filename)
            if not os.path.exists(json_path):
                continue

            with smart_open(json_path) as fin:
                campaign_data = json.load(fin)

            items = campaign_data.get("data", [])

            # Determine reference key based on preference
            chosen_ref_key = None
            for candidate in self.REF_PREFERENCE:
                if any(candidate in s.get("tgt", {}) for row in items for s in row):
                    chosen_ref_key = candidate
                    break

            src_lines = []
            ref_lines = []

            for row in items:
                for s in row:
                    src_raw = s.get("src_text") or s.get("src", "")
                    src_lines.append(self._clean(src_raw))
                    if chosen_ref_key:
                        ref_raw = s.get("tgt", {}).get(chosen_ref_key, "")
                        ref_lines.append(self._clean(ref_raw))

            src_file = self._get_txt_file_path(lp, "src")
            with smart_open(src_file, "wt") as f_src:
                for line in src_lines:
                    print(line, file=f_src)

            if chosen_ref_key:
                ref_file = self._get_txt_file_path(lp, "ref")
                with smart_open(ref_file, "wt") as f_ref:
                    for line in ref_lines:
                        print(line, file=f_ref)
