# VERSION HISTORY

- 1.5.1 (2021-03-05)
  - Fix extraction error for WMT18 extra test sets (test-ts) (#142)
  - Validation and test datasets are added for multilingual TEDx

- 1.5.0 (2021-01-15)
  - Fix an assertion error in chrF (#121)
  - Add missing `__repr__()` methods for BLEU and TER
  - TER: Fix exception when `--short` is used (#131)
  - Pin Mecab version to 1.0.3 for Python 3.5 support
  - [API Change]: Default value for `floor` smoothing is now 0.1 instead of 0.
  - [API Change]: `sacrebleu.sentence_bleu()` now uses the `exp` smoothing method,
    exactly the same as the CLI's --sentence-level behavior. This was mainly done
    to make two methods behave the same.
  - Add smoothing value to BLEU signature (#98)
  - dataset: Fix IWSLT links (#128)
  - Allow variable number of references for BLEU (only via API) (#130).
    Thanks to Ondrej Dusek (@tuetschek)

- 1.4.14 (2020-09-13)
  - Added character-based tokenization (`-tok char`).
    Thanks to Christian Federmann.
  - Added TER (`-m ter`). Thanks to Ales Tamchyna! (fixes #90)
  - Allow calling the script as a standalone utility (fixes #86)
  - Fix type annotation issues (fixes #100) and mark sacrebleu as supporting mypy
  - Added WMT20 robustness test sets:
    - wmt20/robust/set1 (en-ja, en-de)
    - wmt20/robust/set2 (en-ja, ja-en)
    - wmt20/robust/set3 (de-en)

- 1.4.13 (2020-07-30)
  - Added WMT20 newstest test sets (#103)
  - Make mecab3-python an extra dependency, adapt code to new mecab3-python
    This fixes the recent Windows installation issues as well (#104)
    Japanese support should now be explicitly installed through sacrebleu[ja] package.
  - Fix return type annotation of corpus_bleu()
  - Improve sentence_score's documentation, do not allow single ref string (#98)

- 1.4.12 (2020-07-03)
  - Fix a deployment bug (#96)

- 1.4.11 (2020-07-03)
  - Added Multi30k multimodal MT test set metadata
  - Refactored all tokenizers into respective classes (fixes #85)
  - Refactored all metrics into respective classes
  - Moved utility functions into `utils.py`
  - Implemented signatures using `BLEUSignature` and `CHRFSignature` classes
  - Simplified checking of Chinese characters (fixes #5)
  - Unified common regexp tokenization codes for tokenizers (fixes #27)
  - Fixed --detail failing when no test sets are provided
  - Fixed multi-reference BLEU failing when tab-delimited reference stream is used
  - Removed lowercase option for ChrF which was not functional (#85)
  - Simplified ChrF and used the same I/O logic as BLEU to allow for future
    multi-reference reading
  - Added score regression tests for chrF using reference chrF++ implementation
  - Added multi-reference & tokenizer & signature tests

- 1.4.10 (2020-05-30)
  - Fixed bug in signature with mecab tokenizer
  - Cleaned up deprecation warnings (thanks to Karthikeyan Singaravelan @tirkarthi)
  - Now only lists the external [typing](https://pypi.org/project/typing/)
    module as a dependency for Python `<= 3.4`, as it was integrated in the standard
    library in Python 3.5 (thanks to Erwan de Lépinau @ErwanDL).
  - Added LICENSE to pypi (thanks to Mark Harfouche @hmaarrfk)

- 1.4.9 (2020-04-30)
  - Changed `get_available_testsets()` to return a list
  - Remove Japanese MeCab tokenizer from requirements.
    (Must be installed manually to avoid Windows incompatibility).
    Many thanks to Makoto Morishita (@MorinoseiMorizo).

- 1.4.8 (2020-04-26)
  - Added to API:
    - get_source_file()
    - get_reference_files()
    - get_available_testsets()
    - get_langpairs_for_testset()
  - Some internal refactoring
  - Fixed descriptions of some WMT19/google test sets
  - Added API test case (test/test_apy.py)

- 1.4.7 (2020-04-19)
  - Added Google's extra wmt19/en-de refs (-t wmt19/google/{ar,arp,hqall,hqp,hqr,wmtp})
    (Freitag, Grangier, & Caswell
     BLEU might be Guilty but References are not Innocent
     https://arxiv.org/abs/2004.06063)
  - Restored SACREBLEU_DIR and smart_open to exports (thanks to Thomas Liao @tholiao)

- 1.4.6 (2020-03-28)
  - Large internal reorganization as a module (thanks to Thamme Gowda @thammegowda)

- 1.4.5 (2020-03-28)
  - Added Japanese MeCab tokenizer (`-tok ja-mecab`) (thanks to Makoto Morishita @MorinoseiMorizo)
  - Added wmt20/dev test sets (thanks to Martin Popel @martinpopel)

- 1.4.4 (2020-03-10)
  - Smoothing changes (Sebastian Nickels @sn1c)
    - Fixed bug that only applied smoothing to n-grams for n > 2
    - Added default smoothing values for methods "floor" (0) and "add-k" (1)
  - `--list` now returns a list of all language pairs for a task when combined with `-t`
    (e.g., `sacrebleu -t wmt19 --list`)
  - added missing languages for IWSLT17
  - Minor code improvements (Thomas Liao @tholiao)

- 1.4.3 (2019-12-02)
  - Bugfix: handling of result object for CHRF
  - Improved API example

- 1.4.2 (2019-10-11)
  - Tokenization variant omitted from the chrF signature; it is relevant only for BLEU (thanks to Martin Popel)
  - Bugfix: call to sentence_bleu (thanks to Rachel Bawden)
  - Documentation example for Python API (thanks to Vlad Lyalin)
  - Calls to corpus_chrf and sentence_chrf now return a an object instead of a float (use result.score)

- 1.4.1 (2019-09-11)
   - Added sentence-level scoring via -sl (--sentence-level)

- 1.4.0 (2019-09-10)
   - Many thanks to Martin Popel for all the changes below!
   - Added evaluation on concatenated test sets (e.g., `-t wmt17,wmt18`).
     Works as long as they all have the same language pair.
   - Added `sacrebleu --origlang` (both for evaluation on a subset and for `--echo`).
     Note that while echoing prints just the subset, evaluation expects the complete
     test set (and just skips the irrelevant parts).
   - Added `sacrebleu --detail` for breakdown by domain-specific subsets of the test sets.
     (Available for WMT19).
   - Minor changes
     - Improved display of `sacrebleu -h`
     - Added `sacrebleu --list`
     - Code refactoring
     - Documentation and tests updates
     - Fixed a race condition bug (`os.makedirs(outdir, exist_ok=True)` instead of `if os.path.exists`)

- 1.3.7 (2019-07-12)
   - Lazy loading of regexes cuts import time from ~1s to nearly nothing (thanks, @louismartin!)
   - Added a simple (non-atomic) lock on downloading
   - Can now read multiple refs from a single tab-delimited file.
     You need to pass `--num-refs N` to tell it to run the split.
     Only works with a single reference file passed from the command line.

- 1.3.6 (2019-06-10)
   - Removed another f-string for Python 3.5 compatibility

- 1.3.5 (2019-06-07)
   - Restored Python 3.5 compatibility

- 1.3.4 (2019-05-28)
   - Added MTNT 2019 test sets
   - Added a BLEU object

- 1.3.3 (2019-05-08)
   - Added WMT'19 test sets

- 1.3.2 (2018-04-24)
   - Bugfix in test case (thanks to Adam Roberts, @adarob)
   - Passing smoothing method through `sentence_bleu`

- 1.3.1 (2019-03-20)
   - Added another smoothing approach (add-k) and a command-line option for choosing the smoothing method
     (`--smooth exp|floor|add-n|none`) and the associated value (`--smooth-value`), when relevant.
   - Changed interface to some functions (backwards incompatible)
     - 'smooth' is now 'smooth_method'
     - 'smooth_floor' is now 'smooth_value'

- 1.2.21 (19 March 2019)
   - Ctrl-M characters are now treated as normal characters, previously treated as newline.

- 1.2.20 (28 February 2018)
   - Tokenization now defaults to "zh" when language pair is known

- 1.2.19 (19 February 2019)
   - Updated checksum for wmt19/dev (seems to have changed)

- 1.2.18 (19 February 2019)
   - Fixed checksum for wmt17/dev (copy-paste error)

- 1.2.17 (6 February 2019)
   - Added kk-en and en-kk to wmt19/dev

- 1.2.16 (4 February 2019)
   - Added gu-en and en-gu to wmt19/dev

- 1.2.15 (30 January 2019)
   - Added MD5 checksumming of downloaded files for all datasets.

- 1.2.14 (22 January 2019)
   - Added mtnt1.1/train mtnt1.1/valid mtnt1.1/test data from [MTNT](http://www.cs.cmu.edu/~pmichel1/mtnt/)

- 1.2.13 (22 January 2019)
   - Added 'wmt19/dev' task for 'lt-en' and 'en-lt' (development data for new tasks).
   - Added MD5 checksum for downloaded tarballs.

- 1.2.12 (8 November 2018)
   - Now outputs only only digit after the decimal

- 1.2.11 (29 August 2018)
   - Added a function for sentence-level, smoothed BLEU

- 1.2.10 (23 May 2018)
   - Added wmt18 test set (with references)

- 1.2.9 (15 May 2018)
   - Added zh-en, en-zh, tr-en, and en-tr datasets for wmt18/test-ts

- 1.2.8 (14 May 2018)
   - Added wmt18/test-ts, the test sources (only) for [WMT18](http://statmt.org/wmt18/translation-task.html)
   - Moved README out of `sacrebleu.py` and the CHANGELOG into a separate file

- 1.2.7 (10 April 2018)
   - fixed another locale issue (with --echo)
   - grudgingly enabled `-tok none` from the command line

- 1.2.6 (22 March 2018)
   - added wmt17/ms (Microsoft's [additional ZH-EN references](https://github.com/MicrosoftTranslator/Translator-HumanParityData)).
     Try `sacrebleu -t wmt17/ms --cite`.
   - `--echo ref` now pastes together all references, if there is more than one

- 1.2.5 (13 March 2018)
   - added wmt18/dev datasets (en-et and et-en)
   - fixed logic with --force
   - locale-independent installation
   - added "--echo both" (tab-delimited)

- 1.2.3 (28 January 2018)
   - metrics (`-m`) are now printed in the order requested
   - chrF now prints a version string (including the beta parameter, importantly)
   - attempt to remove dependence on locale setting

- 1.2 (17 January 2018)
   - added the chrF metric (`-m chrf` or `-m bleu chrf` for both)
     See 'CHRF: character n-gram F-score for automatic MT evaluation' by Maja Popovic (WMT 2015)
     [http://www.statmt.org/wmt15/pdf/WMT49.pdf]
   - added IWSLT 2017 test and tuning sets for DE, FR, and ZH
     (Thanks to Mauro Cettolo and Marcello Federico).
   - added `--cite` to produce the citation for easy inclusion in papers
   - added `--input` (`-i`) to set input to a file instead of STDIN
   - removed accent mark after objection from UN official

- 1.1.7 (27 November 2017)
   - corpus_bleu() now raises an exception if input streams are different lengths
   - thanks to Martin Popel for:
      - small bugfix in tokenization_13a (not affecting WMT references)
      - adding `--tok intl` (international tokenization)
   - added wmt17/dev and wmt17/dev sets (for languages intro'd those years)

- 1.1.6 (15 November 2017)
   - bugfix for tokenization warning

- 1.1.5 (12 November 2017)
   - added -b option (only output the BLEU score)
   - removed fi-en from list of WMT16/17 systems with more than one reference
   - added WMT16/tworefs and WMT17/tworefs for scoring with both en-fi references

- 1.1.4 (10 November 2017)
   - added effective order for sentence-level BLEU computation
   - added unit tests from sockeye

- 1.1.3 (8 November 2017).
   - Factored code a bit to facilitate API:
      - compute_bleu: works from raw stats
      - corpus_bleu for use from the command line
      - raw_corpus_bleu: turns off tokenization, command-line sanity checks, floor smoothing
   - Smoothing (type 'exp', now the default) fixed to produce mteval-v13a.pl results
   - Added 'floor' smoothing (adds 0.01 to 0 counts, more versatile via API), 'none' smoothing (via API)
   - Small bugfixes, windows compatibility (H/T Christian Federmann)

- 1.0.3 (4 November 2017).
   - Contributions from Christian Federmann:
      - Added explicit support for encoding
      - Fixed Windows support
      - Bugfix in handling reference length with multiple refs

- version 1.0.1 (1 November 2017).
   - Small bugfix affecting some versions of Python.
   - Code reformatting due to Ozan Çağlayan.

- version 1.0 (23 October 2017).
   - Support for WMT 2008--2017.
   - Single tokenization (v13a) with lowercase fix (proper lower() instead of just A-Z).
   - Chinese tokenization.
   - Tested to match all WMT17 scores on all arcs.
