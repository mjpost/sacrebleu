import os
import shutil

import sacrebleu.dataset as dataset
from sacrebleu.utils import smart_open


def test_maybe_download():
    """
    Test the maybe_download function in Dataset class.

    Ensure all raw data files are downloaded and placed in the correct location.
    """
    # ensure all file has been downloaded
    for ds in dataset.DATASETS.values():
        shutil.rmtree(ds._rawdir, ignore_errors=True)
        ds.maybe_download()

        all_file = os.listdir(ds._rawdir)
        for url in ds.data:
            filename = os.path.basename(url)
            assert filename in all_file
            filepath = os.path.join(ds._rawdir, filename)
            assert os.path.getsize(filepath) > 0


def test_process_to_text():
    """
    Test the function `process_to_text` in Dataset class.

    Ensure each field of specified language pair have the same length.
    """
    for ds in dataset.DATASETS.values():
        for filename in os.listdir(ds._outdir):
            filepath = os.path.join(ds._outdir, filename)
            if os.path.isfile(filepath):
                os.remove(filepath)

        ds.process_to_text()

        for pair in ds.langpairs:
            all_files = ds.get_files(pair)

            # count the number of lines in each file
            num_lines = [sum(1 for _ in smart_open(f)) for f in all_files]

            # ensure no empty file
            assert num_lines[0] > 0

            # assert each file has the same length
            assert all(x == num_lines[0] for x in num_lines)


def test_get_files_and_fieldnames():
    """
    Test the functions `get_files` and `fieldnames` in Dataset class.

    Ensure the length of the returned list is correct.
    `get_files()` should return the same number of items as `fieldnames()`.
    """
    for ds in dataset.DATASETS.values():
        for pair in ds.langpairs:
            assert len(ds.get_files(pair)) == len(ds.fieldnames(pair))


def test_source_and_references():
    """
    Test the functions `source` and `references` in Dataset class.

    Ensure the length of source and references are equal.
    """
    for ds in dataset.DATASETS.values():
        for pair in ds.langpairs:
            assert len(list(ds.source(pair))) == len(list(ds.references(pair)))
