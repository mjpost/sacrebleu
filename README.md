# SacréBLEU

SacréBLEU is a standard BLEU implementation that:

- Produces scores on *detokenized* output. 
- Exactly reproduces scores from `mteval-13a.pl`, the official WMT scoring script.
- Automatically downloads WMT datasets, unpacks them, and puts them in plain text format. You just specify the test set name.
- Includes a version string in its output that encapsulates all the parameters, helping with repeatability across papers.

Its goal is to address the following problems in machine translation research:

- BLEU scores computed against differently-tokenized references are not comparable.
- BLEU is a standard metric, but it is subject to a handful of parameters (casing, tokenization, reference length) which can wildly affect the score and which are often not specified in papers.
- Even when specified, it can be hard to dig up the details
- WMT's test sets are released in slightly different formats each year, are wrapped in XML, and are in general a slight bother to deal with.

# Quick start

Get a list of the available test sets:

    ./sbleu    

Download the source for one of the pre-defined test sets:

    ./sbleu -t wmt14 -l de-en --echo src > wmt14-de-en.src

(you can also use long parameter names for readability):

    ./sbleu --test-set wmt14 --langpair de-en --echo src > wmt14-de-en.src

After tokenizing, translating, and then detokenizing it, you can score it easily:

    cat output.detok.txt | ./sbleu -t wmt14 -l de-en

SacréBLEU knows about common WMT test sets, but you can also use it in a backward-compatible mode where you manually specify the reference(s).
It uses the same invocation syntax as Moses' `multi-bleu.txt` or Rico Sennrich's `multi-bleu-detok.perl`:

    ./sbleu -t wmt14 -l de-en --echo ref > wmt14-de-en.ref
    cat ouput.detok.txt | ./sbleu wmt14-de-en.ref

Or, more generally:

    cat output.detok.txt | ./sbleu REF1 [REF2 ...]
    
SacréBLEU generates version strings like the following.
Put them in a footnote in your paper!
Use `--short` for a shorter hash.

    BLEU+case.mixed+lang.de-en+test.wmt17 = 32.97 66.1/40.2/26.6/18.1 (BP = 0.980 ratio = 0.980 hyp_len = 63134 ref_len = 64399)

# Motivation

Comparing BLEU scores is harder than it should be.
Every decoder has its own implementation, offered borrowed from Moses.
Moses itself has a number of implementations as standalone scripts, with little indication of how they differ (note: they mostly don't, but `multi-bleu.pl` expects tokenized output).
Different flags passed to each of these scripts can produce wide swings in the final score.
Other decoders have their own implementations, often borrowed from Moses, but perhaps with subtle changes.
And most importantly, all of these handle tokenization in different ways.
On top of this, downloading and managing test sets is a moderate annoyance.
Sacré bleu!
What a mess.

SacréBLEU aims to solve these problems by wrapping the original Papineni reference implementation together with other useful features.
The defaults are set the way that BLEU should be computed, and furthermore, the script outputs a short version string that allows others to know exactly what you do.
As an added bonus, it automatically downloads and manages test sets for you, so that you can simply tell it to score against 'wmt14', without having to hunt down a path on your local file system.
It is all designed to take BLEU a little more seriously.
After all, even with all its problems, BLEU is default and---admit it---well-loved metric of our entire research community.
Sacré BLEU.

# License

SacréBLEU is licensed under the Apache 2.0 License.

# Credits

This was all Rico Sennrich's idea.

Written by Matt Post, September 2017.
