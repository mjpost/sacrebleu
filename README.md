SacréBLEU provides hassle-free computation of shareable, comparable, and reproducible BLEU scores.
Inspired by Rico Sennrich's `multi-bleu-detok.perl`, it produces the official WMT scores but works with plain text.
It also knows all the standard test sets and handles downloading, processing, and tokenization for you.

Why use this version of BLEU?
- It automatically downloads common WMT test sets and processes them to plain text
- It produces a short version string that facilitates cross-paper comparisons
- It properly computes scores on detokenized outputs, using WMT ([Conference on Machine Translation](http://statmt.org/wmt17)) standard tokenization
- It produces the same values as official script (`mteval-v13a.pl`) used by WMT
- It outputs the BLEU score without the comma, so you don't have to remove it with `sed` (Looking at you, `multi-bleu.perl`)

# QUICK START

Install the Python module (Python 3 only)

    pip3 install sacrebleu

This installs a shell script, `sacrebleu`.
(You can also directly run the shell script `sacrebleu.py` in the source repository).

Get a list of available test sets:

    sacrebleu

Download the source for one of the pre-defined test sets:

    sacrebleu -t wmt14 -l de-en --echo src > wmt14-de-en.src

(you can also use long parameter names for readability):

    sacrebleu --test-set wmt14 --langpair de-en --echo src > wmt14-de-en.src

After tokenizing, translating, and detokenizing it, you can score your decoder output easily:

    cat output.detok.txt | sacrebleu -t wmt14 -l de-en

SacréBLEU knows about common WMT test sets, but you can also use it to score system outputs with arbitrary references.
It also works in backwards compatible model where you manually specify the reference(s), similar to the format of `multi-bleu.txt`:

    cat output.detok.txt | sacrebleu REF1 [REF2 ...]

Note that the system output and references will all be tokenized internally.

SacréBLEU generates version strings like the following.
Put them in a footnote in your paper!
Use `--short` for a shorter hash if you like.

    BLEU+case.mixed+lang.de-en+test.wmt17 = 32.97 66.1/40.2/26.6/18.1 (BP = 0.980 ratio = 0.980 hyp_len = 63134 ref_len = 64399)

# MOTIVATION

Comparing BLEU scores is harder than it should be.
Every decoder has its own implementation, often borrowed from Moses, but maybe with subtle changes.
Moses itself has a number of implementations as standalone scripts, with little indication of how they differ (note: they mostly don't, but `multi-bleu.pl` expects tokenized input).
Different flags passed to each of these scripts can produce wide swings in the final score.
All of these may handle tokenization in different ways.
On top of this, downloading and managing test sets is a moderate annoyance.
Sacré bleu!
What a mess.

SacréBLEU aims to solve these problems by wrapping the original Papineni reference implementation together with other useful features.
The defaults are set the way that BLEU should be computed, and furthermore, the script outputs a short version string that allows others to know exactly what you did.
As an added bonus, it automatically downloads and manages test sets for you, so that you can simply tell it to score against 'wmt14', without having to hunt down a path on your local file system.
It is all designed to take BLEU a little more seriously.
After all, even with all its problems, BLEU is the default and---admit it---well-loved metric of our entire research community.
Sacré BLEU.

# VERSION HISTORY

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

# LICENSE

SacréBLEU is licensed under the Apache 2.0 License.

# CREDITS

This was all Rico Sennrich's idea.
Originally written by Matt Post.
The official version can be found at github.com/awslabs/sockeye, under `contrib/sacrebleu`.
