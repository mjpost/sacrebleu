"""
The base class for all types of datasets.
"""

from typing import Dict, List


class Dataset:
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

    def is_downloaded(self):
        """
        If the dataset isn't downloaded, use utils/download_file()
        This can be implemented here in the base class. It should write
        to ~/.sacreleu/DATASET/raw exactly as it does now.

        :return: True if the file has already been downloaded.
        """
        pass

    def process_to_text(self):
        """
        Class method that essentially does what utils/process_to_text() does.

        This should be implemented by subclasses. Note: process_to_text should write the
        fields in a different format: ~/.sacrebleu/DATASET/DATASET.LANGPAIR.FIELDNAME
        (instead of the current ~/.sacrebleu/DATASET/LANGPAIR.{SRC,REF})
        """
        pass

    def fieldnames(self) -> List[str]:
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
