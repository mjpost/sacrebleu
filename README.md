# sacreBLEU

This repository addresses a couple of problems with score comparisons in the machine translation resarch community:

- Although there is a standard metric, BLEU, it is subject to many parameters, which are often not specified in papers.
- Although there are standard test sets (e.g., those from WMT), they are released in slightly different formats each year, are wrapped in XML, and are in general a slight bother to deal with

To these problems, sacreBLEU provides these solutions:

- BLEU scores are produced on *detokenized* output. sacreBLEU provides its own tokenizations. It includes a version string in its output that encapsulates all the parameters, aiding repeatability.
- It automatically downloads WMT datasets, unpacks them, and puts them in plain text format.

# Quick start

Download the source for one of the pre-defined test sets:

    ./sbleu --test-set wmt14 --langpair de-en --echo src > wmt14-de-en.src

(or use the short flags for faster typing):

    ./sbleu -t wmt14 -l de-en --echo src > wmt14-de-en.src

After tokenizing, translating, and detokenizing it, you can score it easily:

    cat output.detok.txt | ./sbleu -t wmt14 -l de-en

SacreBLEU knows about common test sets, but you can also use it to score system outputs with arbitrary references.
It also works in backwards compatible model where you manually specify the reference(s), similar to the format of `multi-bleu.txt` or Rico Sennrich's `multi-bleu-detok.perl`:

    ./sbleu -t wmt14 -l de-en --echo ref > wmt14-de-en.ref
    cat ouput.detok.txt | ./sbleu wmt14-de-en.ref

Or, more generally:

    cat output.detok.txt | ./sbleu REF1 [REF2 ...]

# In more detail

Computing BLEU scores is a mess.
Every decoder has its own implementation, offered borrowed from Moses.
Moses itself has at least five implementations as standalone scripts, with little indication of how
they differ (multi-bleu, bsbleu, mteval-v11b, mteval-v12, mteval-v13a).
Different flags passed to each of these scripts can drastically affect the final score.
Other decoders have their own implementations, often borrowed from Moses, but perhaps with subtle changes.
And most importantly, all of these handle tokenization in different ways.
Sacre bleu!
What a mess.

SacreBLEU aims to solve these problems by providing a common reference implementation.
The defaults are set the way that BLEU should be computed, and furthermore, the script outputs a short version string that allows others to know exactly what you do.
As an added bonus, it automatically downloads and manages test sets for you, so that you can simply tell it to score against 'wmt14', without having to hunt down a path on your local file system.
It is all designed to take BLEU a little more seriously.
After all, even with all its problems, BLEU is default and---admit it---well-loved metric of an entire research community.
Sacre BLEU.

# License

SacreBLEU is licensed under the Apache 2.0 License.

# Credits

SacreBLEU was Written by Matt Post in September of 2017.
Credit goes to Rico Sennrich, who made the first move in this direction, and various existing BLEU implementations within Moses, which this code repeats.
