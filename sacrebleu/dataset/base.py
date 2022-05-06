"""
The base class for all types of datasets.
"""
import os
import re
from abc import ABCMeta, abstractmethod
from typing import Dict, List

from ..utils import SACREBLEU_DIR, download_file


class Dataset(metaclass=ABCMeta):
    def __init__(
        self,
        name: str,
        data: List[str] = None,
        description: str = None,
        citation: str = None,
        md5: List[str] = None,
        langpairs=Dict[str, List[str]],
        **kwargs,
    ):
        """
        Params come from the values in DATASETS.

        :param name: Name of the dataset.
        :param data: URL of the raw data of the dataset.
        :param description: Description of the dataset.
        :param citation: Citation for the dataset.
        :param md5: MD5 checksum of the dataset.
        :param langpairs: List of available language pairs.
        """
        self.name = name
        self.data = data
        self.description = description
        self.citation = citation
        self.md5 = md5
        self.langpairs = langpairs

        # Don't do any downloading or further processing now.
        # Only do that lazily, when asked.

        # where to store the dataset
        self._outdir = os.path.join(SACREBLEU_DIR, self.name)
        self._rawdir = os.path.join(self._outdir, "raw")

    def maybe_download(self):
        """
        If the dataset isn't downloaded, use utils/download_file()
        This can be implemented here in the base class. It should write
        to ~/.sacreleu/DATASET/raw exactly as it does now.
        """
        os.makedirs(self._rawdir, exist_ok=True)

        expected_checksums = self.md5 if self.md5 else [None] * len(self.data)

        for dataset, expected_md5 in zip(self.data, expected_checksums):
            tarball = os.path.join(self._rawdir, os.path.basename(dataset))

            download_file(
                dataset, tarball, extract_to=self._rawdir, expected_md5=expected_md5
            )

    @staticmethod
    def _clean(s):
        """
        Removes trailing and leading spaces and collapses multiple consecutive internal spaces to a single one.

        :param s: The string.
        :return: A cleaned-up string.
        """
        return re.sub(r"\s+", " ", s.strip())

    def _get_txt_file_path(self, langpair, fieldname):
        """
        Given the language pair and fieldname, return the path to the text file.
        The format is: ~/.sacrebleu/DATASET/DATASET.LANGPAIR.FIELDNAME

        :param langpair: The language pair.
        :param fieldname: The fieldname.
        :return: The path to the text file.
        """
        # handle the special case of subsets. e.g.> "wmt21/dev"
        name = self.name.replace("/", "_")
        return os.path.join(self._outdir, f"{name}.{langpair}.{fieldname}")

    @abstractmethod
    def process_to_text(self):
        """
        Class method that essentially does what utils/process_to_text() does.

        This should be implemented by subclasses. Note: process_to_text should write the
        fields in a different format: ~/.sacrebleu/DATASET/DATASET.LANGPAIR.FIELDNAME
        (instead of the current ~/.sacrebleu/DATASET/LANGPAIR.{SRC,REF})
        """
        pass

    @abstractmethod
    def fieldnames(self) -> List[str]:
        """
        Return a list of all the field names. For most source, this is just
        the source and the reference. For others, it might include the document
        ID for each line, or the original language (origLang).

        get_files() should return the same number of items as this.
        """
        pass

    @abstractmethod
    def __iter__(self):
        """
        Iterates over all fields (source, references, and other metadata) defined
        by the dataset.
        """
        pass

    @abstractmethod
    def source(self):
        """
        Return an iterable over the source lines.
        """
        pass

    @abstractmethod
    def references(self):
        """
        Return an iterable over the references.
        """
        pass

    @abstractmethod
    def get_source_file(self):
        pass

    @abstractmethod
    def get_files(self):
        pass
