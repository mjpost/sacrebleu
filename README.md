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

Get a list of available test sets:

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

Download the source for one of the pre-defined test sets:

```
$ sacrebleu -t wmt17 -l en-de --echo src | head -n2
28-Year-Old Chef Found Dead at San Francisco Mall
A 28-year-old chef who had recently moved to San Francisco was found dead in the stairwell of a local mall this week.

# you can also use long parameter names for readability
$ sacrebleu --test-set wmt17 --language-pair en-de --echo src | head -n2
```

Let's say that you just translated the `en-de` test set of WMT17 with your fancy MT system and the **detokenized** translations are in a file called `output.detok.txt`:

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
$ sacrebleu -t wmt17 -l en-de -i output.detok.txt -f json
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

Let's now compute **multiple metrics** for the same system:

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
    chrF2++|nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.0.0 = 49.0
TER|nrefs:1|case:lc|tok:tercom|norm:no|punct:yes|asian:no|version:2.0.0 = 69.0
```

Metric-specific arguments are detailed in `--help` output:

```
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

SacreBLEU knows about common test sets (as detailed in the `--list` example above), but you can also use it to score system outputs with arbitrary references. In this case, do not forget to provide **detokenized** reference and hypotheses files:

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

## Version Signatures
As you may have noticed, sacreBLEU generates version strings such as `BLEU|nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.0.0` for reproducibility reasons. It's strongly recommended to share these signatures in your papers!

## Translationese Support

If you are interested in the translationese effect, you can evaluate BLEU on a subset of sentences
with a given original language (identified based on the `origlang` tag in the raw SGM files).
E.g., to evaluate only against originally German sentences translated to English use:

    $ sacrebleu -t wmt13 -l de-en --origlang=de -i my-wmt13-output.txt

and to evaluate against the complement (in this case `origlang` en, fr, cs, ru, de) use:

    $ sacrebleu -t wmt13 -l de-en --origlang=non-de -i my-wmt13-output.txt

*Please note* that the evaluator will return a BLEU score only on the requested subset,
but it expects that you pass through the entire translated test set.

## Languages & Preprocessing

### BLEU

- You can compute case-insensitive BLEU by passing `--lowercase` to sacreBLEU
- The default tokenizer for BLEU is `13a` which mimics the `mteval-v13a` script from Moses.
- Other tokenizers are:
   - `none` which will not apply any kind of tokenization at all
   - `char` for language-agnostic character-level tokenization
   - `intl` applies international tokenization and mimics the `mteval-v14` script from Moses
   - `zh` separates out **Chinese** characters and tokenizes the non-Chinese parts using `13a` tokenizer
   - `ja-mecab` tokenizes **Japanese** inputs using the [MeCab](https://pypi.org/project/mecab-python3) morphological analyzer
- You can switch tokenizers using the `--tokenize` flag of sacreBLEU. Alternatively, if you provide language-pair strings
  using `--language-pair/-l`, `zh` and `ja-mecab` tokenizers will be used if the target language is `zh` or `ja`, respectively.
- **Note that** there's no automatic language detection from the hypotheses so you need to make sure that you are correctly
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

### chrF / chrF+

chrF applies minimum to none pre-processing as it deals with character n-grams:

- If you pass `--chrf-whitespace`, whitespace characters will be preserved when computing character n-grams.
- If you pass `--chrf-lowercase`, sacreBLEU will compute case-insensitive chrF(+).
- If you enable chrF+ mode by passing `--chrf-word-order` (The number of `+` letters denotes the word n-gram order that you select),
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

**Pass all references as positional arguments:**
```
$ sacrebleu ref1 ref2 -i system -m bleu chrf ter
        BLEU|nrefs:2|case:mixed|eff:no|tok:13a|smooth:exp|version:2.0.0 = 61.8 <stripped>
      chrF2|nrefs:2|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.0.0 = 75.0
TER|nrefs:2|case:lc|tok:tercom|norm:no|punct:yes|asian:no|version:2.0.0 = 31.2
```

**Alternative (less recommended): Concatenate references using tabs as delimiters:**
```
$ paste ref1 ref2 > refs.tsv

# Don't forget to provide --num-refs/-nr !
$ sacrebleu refs.tsv --num-refs 2 -i system -m bleu
BLEU|nrefs:2|case:mixed|eff:no|tok:13a|smooth:exp|version:2.0.0 = 61.8 <stripped>
```

## Multi-system Evaluation
SacreBLEU supports evaluation of an arbitrary number of systems for a particular
test set and language-pair. This has the advantage of seeing all results in a
nicely formatted table:

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

## Confidence Intervals for Single System Evaluation

- When enabled with the `--confidence` flag, SacreBLEU will print
(1) the actual system score, (2) the true mean estimated from bootstrap resampling and (3),
the 95% [confidence interval](https://en.wikipedia.org/wiki/Confidence_interval) around the mean.

- By default, the number of bootstrap resamples is 2000 (denoted with `bs:2000` in the signature)
and can be changed with `--confidence-n`.

```
$ sacrebleu ref1 -i system -m bleu chrf --confidence -w 4 --short
   BLEU|#:1|bs:2000|rs:12345|c:mixed|e:no|tok:13a|s:exp|v:2.0.0 = 44.5101 (μ = 44.5081 ± 0.724) <stripped>
chrF2|#:1|bs:2000|rs:12345|c:mixed|e:yes|nc:6|nw:0|s:no|v:2.0.0 = 68.8338 (μ = 68.8337 ± 0.496)

# Verify that the first numbers above to the actual scores
$ sacrebleu/sacrebleu.py ref1 -i system -m bleu chrf -w 4 --score-only
44.5101
68.8338
```

**NOTE:** Although provided as a functionality, having access to confidence intervals for just one system
may not reveal much information about the underlying model. It often makes much more sense to perform
**paired statistical tests** using multiple systems.

## Paired Significance Tests for Multi-system Evaluation
Ideally, one would have access to many systems in cases such as (1) investigating
whether a newly added feature yields significantly different scores than the system without that feature or
(2) evaluating submissions for a particular shared task.

SacreBLEU offers two different paired significance tests that are widely used in MT research.

### Paired bootstrap resampling (--paired bs)

- This is an efficient implementation of the paper [Statistical Significance Tests for Machine Translation Evaluation](https://www.aclweb.org/anthology/W04-3250.pdf) and is result-compliant with the [reference Moses implementation](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/analysis/bootstrap-hypothesis-difference-significance.pl).

- Unlike the Moses' implementation that defaults to 1000 bootstrap resamples, SacreBLEU uses a default of 2000 to produce more stable
estimations. This can be changed with the `--paired-n` flag.

- When launched, paired bootstrap resampling will perform:
   - Bootstrap resampling to estimate the 95% CI for all systems and the baseline (similar to `--confidence` for single-system evaluation)
   - A significance test between the **baseline** and each **system** to compute a [p value](https://en.wikipedia.org/wiki/P-value).
  
### Paired approximate randomization (--paired ar)

- Paired approximate randomization (AR) is another type of paired significance test that is claimed to be more accurate than paired bootstrap resampling when it comes to Type-I errors ([Riezler and Maxwell III, 2005](https://www.aclweb.org/anthology/W05-0908.pdf)). Type-I errors
indicate failures to reject the null hypothesis when it is true. In other words, AR should in theory be more robust to subtle changes across systems.

- Our implementation is verified to be result-compliant with the [Multeval toolkit](https://github.com/jhclark/multeval) that also uses paired AR test for pairwise comparison.

- The number of approximate randomization trials is set to 10,000 by default. This can be changed with the `--paired-n` flag.
  
- This method will only compute the p-values for each pairwise comparison. If you also want to get confidence intervals similar to **paired bootstrap resampling**, you need to manually enable it through `--paired-ar-confidence-n <value>`. If `<value>` is 0, the default
of 2000 bootstrap resamples will be used, otherwise `<value>` resamples will be used.

### Running the tests

- The **first system** provided to `--input/-i` will be automatically taken as the **baseline system** against which you want to compare the **other systems.**
The systems will also be automatically named by the provided filename (more specifically, the `basename` of the filenames). (SacreBLEU will automatically discard the baseline system if it appears more than one time. This is a useful trick when you run the tool with something like the following: `-i systems/baseline.txt systems/*.txt`. Here, the `baseline.txt` file will not be also considered as a candidate system.)

- A similar logic applies when tab-separated input file is redirected into SacreBLEU i.e. the first column hypotheses will be taken as the **baseline system**. However, this method is **not recommended** as it won't allow naming your systems in a human-readable way. It will instead enumerate them from 1 to N following the column order in the tab-separated input.

- On Linux and Mac OS X, you can also launch the tests on multiple CPU's by passing the flag `--paired-jobs N`. If equals to 0, SacreBLEU will launch one worker for each pairwise comparison. If > 0, the number of worker processes in the pool will be `N`. This will substantially speed up the runtime especially if you ask **TER** to be computed.

#### Example: Paired bootstrap resampling
In the example below, we set `newstest2017.LIUM-NMT.4900.en-de` as the baseline and compare it to 5 other WMT17 submissions using paired bootstrap resampling. According to the results, the null hypothesis (i.e. the two systems being essentially the same) could not be rejected (at the significance level of 0.05) for the following comparisons:

- 0.3 BLEU difference between the baseline and the FBK system (p = 0.0945)
- 0.1 chrF2 difference between the baseline and the KIT system (p = 0.1089)
- 0.1 BLEU difference between the baseline and the online-B system (p = 0.3073)

```
$ sacrebleu -t wmt17 -l en-de -i newstest2017.LIUM-NMT.4900.en-de newstest2017.* \
        -m bleu chrf --paired bs --quiet
+--------------------------------------------+-----------------------+------------------------+
|                                     System |  BLEU / μ / ± 95% CI  |  chrF2 / μ / ± 95% CI  |
+============================================+=======================+========================+
| Baseline: newstest2017.LIUM-NMT.4900.en-de |  26.6 / 26.6 / 0.65   |   55.9 / 55.9 / 0.47   |
+--------------------------------------------+-----------------------+------------------------+
|                newstest2017.FBK.4870.en-de |  26.3 / 26.3 / 0.65   |   54.7 / 54.7 / 0.48   |
|                                            |     (p = 0.0945)      |     (p = 0.0005)*      |
+--------------------------------------------+-----------------------+------------------------+
|                newstest2017.KIT.4950.en-de |  26.1 / 26.1 / 0.66   |   55.8 / 55.8 / 0.46   |
|                                            |     (p = 0.0105)*     |      (p = 0.1089)      |
+--------------------------------------------+-----------------------+------------------------+
|              newstest2017.online-A.0.en-de |  20.8 / 20.8 / 0.59   |   52.0 / 52.0 / 0.43   |
|                                            |     (p = 0.0005)*     |     (p = 0.0005)*      |
+--------------------------------------------+-----------------------+------------------------+
|              newstest2017.online-B.0.en-de |  26.7 / 26.7 / 0.67   |   56.3 / 56.3 / 0.45   |
|                                            |     (p = 0.3073)      |     (p = 0.0240)*      |
+--------------------------------------------+-----------------------+------------------------+
|   newstest2017.PROMT-Rule-based.4735.en-de |  16.6 / 16.6 / 0.51   |   50.4 / 50.4 / 0.40   |
|                                            |     (p = 0.0005)*     |     (p = 0.0005)*      |
+--------------------------------------------+-----------------------+------------------------+

------------------------------------------------------------
Paired bootstrap resampling test with 2000 resampling trials
------------------------------------------------------------
 - Each system is pairwise compared to Baseline: newstest2017.LIUM-NMT.4900.en-de.
   Actual system score / estimated true mean / 95% CI are provided for each metric.

 - Null hypothesis: the system and the baseline translations are essentially
   generated by the same underlying process. The p-value is roughly the probability
   of the absolute score difference (delta) between a system and the {bline} occurring due to chance.

 - Assuming a significance threshold of 0.05, the Null hypothesis can be rejected
   for p-values < 0.05 (marked with "*"). This means that the delta is unlikely to be attributed
   to chance, hence the system is significantly "different" than the baseline.
   Otherwise, the p-values are highlighted in red (if the terminal supports colors).

 - NOTE: Significance does not tell whether a system is "better" than the baseline but rather
   emphasizes the "difference" of the systems in terms of the replicability of the delta.

-----------------
Metric signatures
-----------------
 - BLEU       nrefs:1|bs:2000|seed:12345|case:mixed|eff:no|tok:13a|smooth:exp|version:2.0.0
 - chrF2      nrefs:1|bs:2000|seed:12345|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.0.0
```

#### Example: Paired approximate randomization

Let's now run the paired approximate randomization test for the same systems. According to the results, the findings are compatible with the paired bootstrap resampling test. However, the p-values here are much more higher (i.e. the test is much more confident that the deltas between the baseline and the FBK/KIT/online-B systems are due to chance). 

```
$ sacrebleu -t wmt17 -l en-de -i newstest2017.LIUM-NMT.4900.en-de newstest2017.* \
        -m bleu chrf --paired ar --quiet
+--------------------------------------------+---------------+---------------+
|                                     System |     BLEU      |     chrF2     |
+============================================+===============+===============+
| Baseline: newstest2017.LIUM-NMT.4900.en-de |     26.6      |     55.9      |
+--------------------------------------------+---------------+---------------+
|                newstest2017.FBK.4870.en-de |     26.3      |     54.7      |
|                                            | (p = 0.1887)  | (p = 0.0001)* |
+--------------------------------------------+---------------+---------------+
|                newstest2017.KIT.4950.en-de |     26.1      |     55.8      |
|                                            | (p = 0.0193)* | (p = 0.2511)  |
+--------------------------------------------+---------------+---------------+
|              newstest2017.online-A.0.en-de |     20.8      |     52.0      |
|                                            | (p = 0.0001)* | (p = 0.0001)* |
+--------------------------------------------+---------------+---------------+
|              newstest2017.online-B.0.en-de |     26.7      |     56.3      |
|                                            | (p = 0.8066)  | (p = 0.0385)* |
+--------------------------------------------+---------------+---------------+
|   newstest2017.PROMT-Rule-based.4735.en-de |     16.6      |     50.4      |
|                                            | (p = 0.0001)* | (p = 0.0001)* |
+--------------------------------------------+---------------+---------------+

 <stripped>
```

# Using SacreBLEU from Python

For evaluation, it may be useful to compute BLEU, chrF or TER from a Python script. The recommended
way of doing this is to use the object-oriented API, by creating an instance of the `metrics.BLEU` class
for example:

```python
In [1]: from sacrebleu.metrics import BLEU, CHRF, TER
   ...: 
   ...: refs = [ # First set of references
   ...:          ['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.'],
   ...:          # Second set of references
   ...:          ['The dog had bit the man.', 'No one was surprised.', 'The man had bitten the dog.'],
   ...:        ]
   ...: sys = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']

In [2]: bleu = BLEU()

In [3]: bleu.corpus_score(sys, refs)
Out[3]: BLEU = 48.53 82.4/50.0/45.5/37.5 (BP = 0.943 ratio = 0.944 hyp_len = 17 ref_len = 18)

In [4]: bleu.get_signature()
Out[4]: nrefs:2|case:mixed|eff:no|tok:13a|smooth:exp|version:2.0.0

In [5]: chrf = CHRF()

In [6]: chrf.corpus_score(sys, refs)
Out[6]: chrF2 = 59.73
```

### Variable Number of References

Let's now remove the first reference sentence for the first system sentence `The dog bit the man.` by replacing it with either `None` or the empty string `''`.
This allows using a variable number of reference segments per hypothesis. Observe how the signature changes from `nrefs:2` to `nrefs:var`:

```python
In [1]: from sacrebleu.metrics import BLEU, CHRF, TER
   ...: 
   ...: refs = [ # First set of references
                 # 1st sentence does not have a ref here
   ...:          ['', 'It was not unexpected.', 'The man bit him first.'],
   ...:          # Second set of references
   ...:          ['', 'No one was surprised.', 'The man had bitten the dog.'],
   ...:        ]
   ...: sys = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']
   
In [2]: bleu = BLEU()

In [3]: bleu.corpus_score(sys, refs)
Out[3]: BLEU = 29.44 82.4/42.9/27.3/12.5 (BP = 0.889 ratio = 0.895 hyp_len = 17 ref_len = 19)

In [4]: bleu.get_signature()
Out[4]: nrefs:var|case:mixed|eff:no|tok:13a|smooth:exp|version:2.0.0
```

## Compatibility API

You can also use the compatibility API that provides wrapper functions around the object-oriented API to
compute sentence-level and corpus-level BLEU, chrF and TER: (It should be noted that this API can be
removed in future releases)

```python
In [1]: import sacrebleu
   ...: 
   ...: refs = [ # First set of references
   ...:          ['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.'],
   ...:          # Second set of references
   ...:          ['The dog had bit the man.', 'No one was surprised.', 'The man had bitten the dog.'],
   ...:        ]
   ...: sys = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']

In [2]: sacrebleu.corpus_bleu(sys, refs)
Out[2]: BLEU = 48.53 82.4/50.0/45.5/37.5 (BP = 0.943 ratio = 0.944 hyp_len = 17 ref_len = 18)
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
