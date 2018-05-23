# VERSION HISTORY

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
