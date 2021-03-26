# sacreBLEU

[![PyPI version](https://img.shields.io/pypi/v/sacrebleu)](https://img.shields.io/pypi/v/sacrebleu)
[![Python version](https://img.shields.io/pypi/pyversions/sacrebleu)](https://img.shields.io/pypi/pyversions/sacrebleu)
[![GitHub issues](https://img.shields.io/github/issues/mjpost/sacreBLEU.svg)](https://github.com/mjpost/sacrebleu/issues)

SacreBLEU ([Post, 2018](http://aclweb.org/anthology/W18-6319)) provides hassle-free computation of shareable, comparable, and reproducible **BLEU** scores.
Inspired by Rico Sennrich's `multi-bleu-detok.perl`, it produces the official WMT scores but works with plain text.
It also knows all the standard test sets and handles downloading, processing, and tokenization for you.

In recent versions, sacreBLEU also provides:

- Different tokenizers for BLEU including support for Japanese and Chinese
- Support for **chrF, chrF++** and **Translation error rate (TER)** metrics
- An object-oriented API for all metrics
- Paired bootstrap resampling and approximate randomization tests for significance

The official version is hosted at <https://github.com/mjpost/sacrebleu>.

# Motivation

Comparing BLEU scores is harder than it should be. Every decoder has its own implementation, often borrowed from Moses, but maybe with subtle changes.
Moses itself has a number of implementations as standalone scripts, with little indication of how they differ (note: they mostly don't, but `multi-bleu.pl` expects tokenized input). Different flags passed to each of these scripts can produce wide swings in the final score. All of these may handle tokenization in different ways. On top of this, downloading and managing test sets is a moderate annoyance.

Sacre bleu! What a mess.

**SacreBLEU** aims to solve these problems by wrapping the original reference implementation ([Papineni et al., 2002](https://www.aclweb.org/anthology/P02-1040.pdf)) together with other useful features.
The defaults are set the way that BLEU should be computed, and furthermore, the script outputs a short version string that allows others to know exactly what you did.
As an added bonus, it automatically downloads and manages test sets for you, so that you can simply tell it to score against `wmt14`, without having to hunt down a path on your local file system.
It is all designed to take BLEU a little more seriously.
After all, even with all its problems, BLEU is the default and---admit it---well-loved metric of our entire research community.
Sacre BLEU.

# Features

- It automatically downloads common WMT test sets and processes them to plain text
- It produces a short version string that facilitates cross-paper comparisons
- It properly computes scores on detokenized outputs, using WMT ([Conference on Machine Translation](http://statmt.org/wmt17)) standard tokenization
- It produces the same values as official script (`mteval-v13a.pl`) used by WMT
- It outputs the BLEU score without the comma, so you don't have to remove it with `sed` (Looking at you, `multi-bleu.perl`)

# Installation

Install the official Python module from PyPI (**Python>=3.6 only**):

    pip install sacrebleu

In order to install Japanese tokenizer support through `mecab-python3`, you need to run the
following command instead, to perform a full installation with dependencies:

    pip install sacrebleu[ja]

# Command-line Usage

## Basics

- Get a list of available test sets:

```
$ sacrebleu --list
iwslt17                       : Official evaluation data for IWSLT.
iwslt17/dev2010               : Development data for IWSLT 2017.
iwslt17/tst2010               : Development data for IWSLT 2017.
iwslt17/tst2011               : Development data for IWSLT 2017.
iwslt17/tst2012               : Development data for IWSLT 2017.
iwslt17/tst2013               : Development data for IWSLT 2017.
iwslt17/tst2014               : Development data for IWSLT 2017.
iwslt17/tst2015               : Development data for IWSLT 2017.
iwslt17/tst2016               : Development data for IWSLT 2017.
mtnt1.1/test                  : Test data for the Machine Translation of Noisy Text
mtnt1.1/train                 : Training data for the Machine Translation of Noisy Text
mtnt1.1/valid                 : Validation data for the Machine Translation of Noisy Text
mtnt2019                      : Test set for the WMT 19 robustness shared task
multi30k/2016                 : 2016 flickr test set of Multi30k dataset
multi30k/2017                 : 2017 flickr test set of Multi30k dataset
multi30k/2018                 : 2018 flickr test set of Multi30k dataset
wmt08                         : Official evaluation data.
wmt08/europarl                : Official evaluation data (Europarl).
wmt08/nc                      : Official evaluation data (news commentary).
wmt09                         : Official evaluation data.
wmt10                         : Official evaluation data.
wmt11                         : Official evaluation data.
wmt12                         : Official evaluation data.
wmt13                         : Official evaluation data.
wmt14                         : Official evaluation data.
wmt14/full                    : Evaluation data released after official evaluation for further research.
wmt15                         : Official evaluation data.
wmt16                         : Official evaluation data.
wmt16/B                       : Additional reference for EN-FI.
wmt16/dev                     : Development sets released for new languages in 2016.
wmt16/tworefs                 : EN-FI with two references.
wmt17                         : Official evaluation data.
wmt17/B                       : Additional reference for EN-FI and FI-EN.
wmt17/dev                     : Development sets released for new languages in 2017.
wmt17/improved                : Improved zh-en and en-zh translations.
wmt17/ms                      : Additional Chinese-English references from Microsoft Research.
wmt17/tworefs                 : Systems with two references.
wmt18                         : Official evaluation data.
wmt18/dev                     : Development data (Estonian<>English).
wmt18/test-ts                 : Official evaluation sources with extra test sets interleaved.
wmt19                         : Official evaluation data.
wmt19/dev                     : Development data for tasks new to 2019.
wmt19/google/ar               : Additional high-quality reference for WMT19/en-de.
wmt19/google/arp              : Additional paraphrase of wmt19/google/ar.
wmt19/google/hqall            : Best human-selected reference among original official reference and the Google reference and paraphrases.
wmt19/google/hqp              : Best human-selected reference between wmt19/google/arp and wmt19/google/wmtp.
wmt19/google/hqr              : Best human selected-reference between wmt19 and wmt19/google/ar.
wmt19/google/wmtp             : Additional paraphrase of the official WMT19 reference.
wmt20                         : Official evaluation data for WMT20
wmt20/dev                     : Development data for tasks new to 2020.
wmt20/robust/set1             : WMT20 robustness task, set 1
wmt20/robust/set2             : WMT20 robustness task, set 2
wmt20/robust/set3             : WMT20 robustness task, set 3
wmt20/tworefs                 : WMT20 news test sets with two references
```

- Download the source for one of the pre-defined test sets:
```
$ sacrebleu -t wmt17 -l en-de --echo src | head -n2
28-Year-Old Chef Found Dead at San Francisco Mall
A 28-year-old chef who had recently moved to San Francisco was found dead in the stairwell of a local mall this week.

# you can also use long parameter names for readability
$ sacrebleu --test-set wmt17 --language-pair en-de --echo src | head -n2
```

- Let's say that you just translated the `en-de` test set of WMT17 with your fancy MT system and the **detokenized** translations
  are in a file called `output.detok.txt`:
```
# Option 1: Redirect system output to STDIN
$ cat output.detok.txt | sacrebleu -t wmt17 -l en-de
BLEU|nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.0.0 = 20.8 54.4/26.6/14.9/8.7 (BP = 1.000 ratio = 1.026 hyp_len = 62880 ref_len = 61287)

# Option 2: Using the --input/-i argument
$ sacrebleu -t wmt17 -l en-de -i output.detok.txt
BLEU|nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.0.0 = 20.8 54.4/26.6/14.9/8.7 (BP = 1.000 ratio = 1.026 hyp_len = 62880 ref_len = 61287)

# You can obtain a short version of the signature with --short/-sh
$ sacrebleu -t wmt17 -l en-de -i output.detok.txt -sh
BLEU|#:1|c:mixed|e:no|tok:13a|s:exp|v:2.0.0 = 20.8 54.4/26.6/14.9/8.7 (BP = 1.000 ratio = 1.026 hyp_len = 62880 ref_len = 61287)

# You can also let it print the score only with --score-only/-b
$ sacrebleu -t wmt17 -l en-de -i output.detok.txt -b
20.8

# Add more precision to the score with --width/-w
$ sacrebleu -t wmt17 -l en-de -i output.detok.txt -b -w 4
20.7965

# Finally you can dump the information as JSON with --format/-f json
$ sacrebleu -t wmt17 -l en-de -i output.detok.txt
{
 "name": "BLEU",
 "score": 20.796506153855994,
 "signature": "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.0.0",
 "verbose_score": "54.4/26.6/14.9/8.7 (BP = 1.000 ratio = 1.026 hyp_len = 62880 ref_len = 61287)",
 "nrefs": "1",
 "case": "mixed",
 "eff": "no",
 "tok": "13a",
 "smooth": "exp",
 "version": "2.0.0"
}
```

- Let's now compute multiple metrics for the same system:

```
# Let's first compute BLEU, chrF and TER with the default settings
$ sacrebleu -t wmt17 -l en-de -i output.detok.txt -m bleu chrf ter
        BLEU|nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.0.0 = 20.8 <stripped>
      chrF2|nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.0.0 = 52.0
TER|nrefs:1|case:lc|tok:tercom|norm:no|punct:yes|asian:no|version:2.0.0 = 69.0

# Let's enable chrF++ which is a revised version of chrF that takes into account word n-grams
# Observe how the nw:0 gets changed into nw:2 for chrF in line 2
$ sacrebleu -t wmt17 -l en-de -i output.detok.txt -m bleu chrf ter --chrf-word-order 2
        BLEU|nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.0.0 = 20.8 <stripped>
      chrF2|nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.0.0 = 49.0
TER|nrefs:1|case:lc|tok:tercom|norm:no|punct:yes|asian:no|version:2.0.0 = 69.0

# Metric-specific arguments are detailed in --help
BLEU related arguments:
  --smooth-method {none,floor,add-k,exp}, -s {none,floor,add-k,exp}
                        Smoothing method: exponential decay, floor (increment
                        zero counts), add-k (increment num/denom by k for
                        n>1), or none. (Default: exp)
  --smooth-value BLEU_SMOOTH_VALUE, -sv BLEU_SMOOTH_VALUE
                        The smoothing value. Only valid for floor and add-k.
                        (Defaults: floor: 0.1, add-k: 1)
  --tokenize {none,zh,13a,char,intl,ja-mecab}, -tok {none,zh,13a,char,intl,ja-mecab}
                        Tokenization method to use for BLEU. If not provided,
                        defaults to `zh` for Chinese, `ja-mecab` for Japanese
                        and `13a` (mteval) otherwise.
  --lowercase, -lc      If True, enables case-insensitivity. (Default: False)
  --force               Insist that your tokenized input is actually
                        detokenized.

chrF related arguments:
  --chrf-char-order CHRF_CHAR_ORDER, -cc CHRF_CHAR_ORDER
                        Character n-gram order. (Default: 6)
  --chrf-word-order CHRF_WORD_ORDER, -cw CHRF_WORD_ORDER
                        Word n-gram order (Default: 0). If equals to 2, the
                        metric is referred to as chrF++.
  --chrf-beta CHRF_BETA
                        Determine the importance of recall w.r.t precision.
                        (Default: 2)
  --chrf-whitespace     Include whitespaces when extracting character n-grams.
                        (Default: False)
  --chrf-lowercase      Enable case-insensitivity. (Default: False)
  --chrf-eps-smoothing  Enables epsilon smoothing similar to chrF++.py, NLTK
                        and Moses; instead of effective order smoothing.
                        (Default: False)

TER related arguments (The defaults replicate TERCOM's behavior):
  --ter-case-sensitive  Enables case sensitivity (Default: False)
  --ter-asian-support   Enables special treatment of Asian characters
                        (Default: False)
  --ter-no-punct        Removes punctuation. (Default: False)
  --ter-normalized      Applies basic normalization and tokenization.
                        (Default: False)
```

- SacreBLEU knows about common test sets (as detailed in the `--list` example above), but you can also use it to score system outputs with arbitrary references. In this case, do not forget to provide **detokenized** reference and hypotheses files:

```
# Let's save the reference to a text file
$ sacrebleu -t wmt17 -l en-de --echo ref > ref.detok.txt

# Option 1: Pass the reference file as a positional argument to sacreBLEU
$ sacrebleu ref.detok.txt -i output.detok.txt -m bleu -b -w 4
20.7965

# Option 2: Redirect the system into STDIN (Compatible with multi-bleu.perl way of doing things)
$ cat output.detok.txt | sacrebleu ref.detok.txt -m bleu -b -w 4
20.7965
```

## Languages & Preprocessing

### BLEU

- You can compute case-insensitive BLEU by passing `--lowercase` to sacreBLEU
- The default tokenizer for BLEU is `13a` which mimics the `mteval-v13a` script from Moses.
- Other options are:
   - `none` which will not apply any kind of tokenization at all
   - `char` for language-agnostic character-level tokenization
   - `intl` applies international tokenization and mimics the `mteval-v14` script from Moses
   - `zh` separates out **Chinese** characters and tokenizes the non-Chinese parts using `13a` tokenizer
   - `ja-mecab` tokenizes **Japanese** inputs using the [MeCab](https://pypi.org/project/mecab-python3) morphological analyzer
- You can switch tokenizers using the `--tokenize` flag of sacreBLEU. Alternatively, if you provide language-pair strings
  using `--language-pair/-l`, `zh` and `ja-mecab` tokenizers will be used if the target language is `zh` or `ja`, respectively.
- **NOTE:** There's no automatic language detection from the hypotheses so you need to make sure that you are correctly
  selecting the tokenizer for **Japanese** and **Chinese**.

```
# Default 13a tokenizer will produce poor results for Japanese
$ sacrebleu kyoto-test.ref.ja -i kyoto-test.hyp.ja -b
2.1

# Use the ja-mecab tokenizer
$ sacrebleu kyoto-test.ref.ja -i kyoto-test.hyp.ja --tokenize ja-mecab -b
14.5

# Alternatively, if you provide the language-pair, sacreBLEU will use ja-mecab automatically
$ sacrebleu kyoto-test.ref.ja -i kyoto-test.hyp.ja -l en-ja -b
14.5
```

### chrF

chrF does a minimum effort on tokenization since it deals with character n-grams.
- If you pass `--chrf-whitespace`, whitespace characters will be preserved when computing the character n-grams.
- If you pass `--chrf-lowercase`, sacreBLEU will compute case-insensitive chrF(+).
- If you enable chrF+ mode by passing `--chrf-word-order` (Number of `+` letters denotes the word n-gram order that you select),
  a very simple punctuation tokenization will be internally applied.


### TER

Translation Error Rate (TER) has its own special tokenizer that you can configure through the command line.
The defaults provided are **compatible with the upstream TER implementation (TERCOM)** but you can nevertheless modify the
behavior through the command-line:
- TER is by default case-insensitive. Pass `--ter-case-sensitive` to enable case-sensitivity.
- Pass `--ter-normalize` to apply a general Western tokenization
- Pass `--ter-asian-support` to enable the tokenization of Asian characters. If provided with `--ter-normalize`,
  both will be applied.
- Pass `--ter-no-punct` to strip punctuation.

## Multi-reference Evaluation

All three metrics support the use of multiple references during evaluation:

```
# Single reference: ref1
$ sacrebleu ref1 -i system -m bleu chrf ter
        BLEU|nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.0.0 = 44.5 <stripped>
      chrF2|nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.0.0 = 68.8
TER|nrefs:1|case:lc|tok:tercom|norm:no|punct:yes|asian:no|version:2.0.0 = 40.4

# Pass each reference as positional arguments
$ sacrebleu ref1 ref2 -i system -m bleu chrf ter
        BLEU|nrefs:2|case:mixed|eff:no|tok:13a|smooth:exp|version:2.0.0 = 61.8 <stripped>
      chrF2|nrefs:2|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.0.0 = 75.0
TER|nrefs:2|case:lc|tok:tercom|norm:no|punct:yes|asian:no|version:2.0.0 = 31.2

# Alternative (less recommended): Paste each system using '\t' delimited lines
$ paste ref1 ref2 > refs.tsv

# Don't forget to provide --num-refs/-nr !
$ sacrebleu refs.tsv --num-refs 2 -i system -m bleu
BLEU|nrefs:2|case:mixed|eff:no|tok:13a|smooth:exp|version:2.0.0 = 61.8 <stripped>
```

## Multi-system Evaluation
sacreBLEU supports evaluation of an arbitrary number of systems for a particular
test set and language-pair. This has the advantage of seeing all results in a
nicely formatted table.

```
# Pass multiple systems to --input/-i
$ sacrebleu -t wmt17 -l en-de -i newstest2017.online-* -m bleu chrf
sacreBLEU: Found 4 systems.
+-------------------------------+--------+---------+
|                        System |  BLEU  |  chrF2  |
+===============================+========+=========+
| newstest2017.online-A.0.en-de |  20.8  |  52.0   |
+-------------------------------+--------+---------+
| newstest2017.online-B.0.en-de |  26.7  |  56.3   |
+-------------------------------+--------+---------+
| newstest2017.online-F.0.en-de |  15.5  |  49.3   |
+-------------------------------+--------+---------+
| newstest2017.online-G.0.en-de |  18.2  |  51.6   |
+-------------------------------+--------+---------+

-----------------
Metric signatures
-----------------
 - BLEU       nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.0.0
 - chrF2      nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.0.0
```

You can also change the output format from `text` to one of `latex, rst, html`:
```
# Prints a LaTeX table
$ sacrebleu -t wmt17 -l en-de -i newstest2017.online-* -m bleu chrf -f latex
sacreBLEU: Found 4 systems.
\begin{tabular}{rcc}
\toprule
                        System &  BLEU  &  chrF2  \\
\midrule
 newstest2017.online-A.0.en-de &  20.8  &  52.0   \\
 newstest2017.online-B.0.en-de &  26.7  &  56.3   \\
 newstest2017.online-F.0.en-de &  15.5  &  49.3   \\
 newstest2017.online-G.0.en-de &  18.2  &  51.6   \\
\bottomrule
\end{tabular}

-----------------
Metric signatures
-----------------
 - BLEU       nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.0.0
 - chrF2      nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.0.0
```

## Version Signatures
As you may have noticed, sacreBLEU generates version strings such as `BLEU|nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.0.0` for reproducibility reasons. It's strongly recommended to share these signatures in your papers!

## Translationese Support

If you are interested in the translationese effect, you can evaluate BLEU on a subset of sentences
with a given original language (identified based on the `origlang` tag in the raw SGM files).
E.g., to evaluate only against originally German sentences translated to English use:

    sacrebleu -t wmt13 -l de-en --origlang=de < my-wmt13-output.txt

and to evaluate against the complement (in this case `origlang` en, fr, cs, ru, de) use:

    sacrebleu -t wmt13 -l de-en --origlang=non-de < my-wmt13-output.txt

*Please note* that the evaluator will return a BLEU score only on the requested subset,
but it expects that you pass through the entire translated test set.

# Using SacreBLEU from Python

## Compatibility API

For evaluation, it may be useful to compute BLEU inside a script. This is how you can do it:
```python
import sacrebleu
refs = [['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.'],
        ['The dog had bit the man.', 'No one was surprised.', 'The man had bitten the dog.']]
sys = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']
bleu = sacrebleu.corpus_bleu(sys, refs)
print(bleu.score)
```

# License

SacreBLEU is licensed under the [Apache 2.0 License](LICENSE.txt).

# Credits

This was all Rico Sennrich's idea.
Originally written by Matt Post.
New features and ongoing support provided by Martin Popel (@martinpopel) and Ozan Caglayan (@ozancaglayan).

If you use SacreBLEU, please cite the following:

```
@inproceedings{post-2018-call,
  title = "A Call for Clarity in Reporting {BLEU} Scores",
  author = "Post, Matt",
  booktitle = "Proceedings of the Third Conference on Machine Translation: Research Papers",
  month = oct,
  year = "2018",
  address = "Belgium, Brussels",
  publisher = "Association for Computational Linguistics",
  url = "https://www.aclweb.org/anthology/W18-6319",
  pages = "186--191",
}
```

# Release Notes

Please see [CHANGELOG.md](CHANGELOG.md) for release notes.
