import os
import random
import shutil

import sacrebleu.dataset as dataset
from sacrebleu.utils import smart_open


def test_maybe_download():
    """
    Test the maybe_download function in Dataset class.

    Check a few random datasets for downloading and correct file placement.
    """
    # ensure all file have been downloaded
    selected_datasets = random.choices(list(dataset.DATASETS.values()), k=10)
    for ds in selected_datasets:
        shutil.rmtree(ds._rawdir, ignore_errors=True)
        ds.maybe_download()

        all_files = os.listdir(ds._rawdir)
        for url in ds.data:
            filename = ds._get_tarball_filename(url)
            assert filename in all_files
            filepath = os.path.join(ds._rawdir, filename)
            assert os.path.getsize(filepath) > 0


def test_process_to_text():
    """
    Test the function `process_to_text` in Dataset class.

    Ensure each field of specified language pair have the same length.
    """
    selected_datasets = random.choices(list(dataset.DATASETS.values()), k=10)
    for ds in selected_datasets:
        if os.path.exists(ds._outdir):
            for filename in os.listdir(ds._outdir):
                filepath = os.path.join(ds._outdir, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)

        ds.process_to_text()

        for pair in ds.langpairs:
            all_files = ds.get_files(pair)
            source_file = ds.get_source_file(pair)
            reference_files = ds.get_reference_files(pair)
            fields = ds.fieldnames(pair)

            source = smart_open(source_file).readlines()
            references = [smart_open(f).readlines() for f in reference_files]
            all_ = [smart_open(f).readlines() for f in all_files]

            len_src = len(source)
            len_ref = len(references[0])
            len_aligned = ds.doc_align(pair, source, "src")
            for lines, field in zip(all_, fields):
                assert (
                    len(lines) == len_src or len(lines) == len_ref
                ), f"{ds}: {pair}: {field} has different length with both src and refs."

                if field != "src":
                    assert (
                        ds.doc_align(pair, lines, "ref") == len_aligned
                    ), f"{ds}: {pair}: {field} can't be aligned with src."


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
            src_len = len(list(ds.source(pair)))
            ref_len = len(list(ds.references(pair)))
            assert (
                src_len == ref_len
            ), f"source/reference failure for {ds.name}:{pair} len(source)={src_len} len(references)={ref_len}"
            source = ds.source(pair)
            aligned_source = ds.doc_align(pair, source, "src")
            references = ds.references(pair)
            aligned_references = ds.doc_align(pair, references, "ref")
            assert len(aligned_source) == len(aligned_references)


def test_wmt22_references():
    """
    WMT21 added the ability to specify which reference to use (among many in the XML).
    The default was "A" for everything.
    WMT22 added the ability to override this default on a per-langpair basis, by
    replacing the langpair list of paths with a dict that had the list of paths and
    the annotator override.
    """
    wmt22 = dataset.DATASETS["wmt22"]

    # make sure CS-EN returns all reference fields
    cs_en_fields = wmt22.fieldnames("cs-en")
    for ref in ["ref:B", "ref:C"]:
        assert ref in cs_en_fields
    assert "ref:A" not in cs_en_fields

    # make sure ref:B is the one used by default
    assert wmt22._get_langpair_allowed_refs("cs-en") == ["ref:B"]

    # similar check for another dataset: there should be no default ("A"),
    # and the only ref found should be the unannotated one
    assert "ref:A" not in wmt22.fieldnames("liv-en")
    assert "ref" in wmt22.fieldnames("liv-en")

    # and that ref:A is the default for all languages where it wasn't overridden
    for langpair, langpair_data in wmt22.langpairs.items():
        if type(langpair_data) == dict:
            assert wmt22._get_langpair_allowed_refs(langpair) != ["ref:A"]
        else:
            assert wmt22._get_langpair_allowed_refs(langpair) == ["ref:A"]
