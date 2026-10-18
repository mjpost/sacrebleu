from __future__ import annotations

import json
import os
from collections import defaultdict

from ..utils import smart_open
from .base import Dataset


class LTBDataset(Dataset):
    """
    The Last Translation Benchmark dataset (https://huggingface.co/datasets/zouhar/last-translation-benchmark).
    Filters items by tag (default: 'LTBv1-eval') and extracts the 'human' translation as reference.
    """

    def __init__(
        self,
        name: str,
        data: list[str] | None = None,
        description: str | None = None,
        citation: str | None = None,
        md5: list[str] | None = None,
        langpairs=None,
        tag: str = "LTBv1-eval",
        **kwargs,
    ):
        super().__init__(name, data, description, citation, md5, langpairs, **kwargs)
        self.tag = tag

    def process_to_text(self, langpair=None) -> None:
        """Processes raw JSON dataset to plain text files for all language pairs."""
        self.maybe_download()
        self._get_langpair_metadata(langpair)

        raw_file = os.path.join(self._rawdir, self._get_tarball_filename(self.data[0]))
        with smart_open(raw_file) as fin:
            data = json.load(fin)

        lines_by_lp: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: {"src": [], "ref": []}
        )

        for item in data:
            if self.tag not in item.get("tags", []):
                continue

            src_iso = item.get("source_lang_iso")
            tgt_iso = item.get("target_lang_iso")
            if not src_iso or not tgt_iso:
                continue

            lp = f"{src_iso}-{tgt_iso}"
            if lp not in self.langpairs:
                continue

            human_ref = None
            for trans in item.get("translations", []):
                if trans.get("model") == "human":
                    human_ref = trans.get("translation")
                    break

            if human_ref is None:
                continue

            src_text = self._clean(item.get("source_text", ""))
            ref_text = self._clean(human_ref)

            lines_by_lp[lp]["src"].append(src_text)
            lines_by_lp[lp]["ref"].append(ref_text)

        os.makedirs(self._outdir, exist_ok=True)
        for lp, lines in lines_by_lp.items():
            src_file = self._get_txt_file_path(lp, "src")
            ref_file = self._get_txt_file_path(lp, "ref")
            with smart_open(src_file, "wt") as f_src:
                for line in lines["src"]:
                    print(line, file=f_src)
            with smart_open(ref_file, "wt") as f_ref:
                for line in lines["ref"]:
                    print(line, file=f_ref)
