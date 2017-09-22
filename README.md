# The problem

Computing BLEU scores is a huge mess.
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

# Quick start

BLEU scores should be computed on *detokenized* system outputs.
`sbleu` therefore expects that you will do so beforehand.
It applies its own tokenization to both your system output and the reference.
This fact that the reference itself is tokenized the same way is what makes scores comparable across papers.

Here is an example usage.

    $ cat output.detok.de | sbleu -t wmt14

# License

SacreBLEU is licensed under the Apache 2.0 License.

# Credits

SacreBLEU was Written by Matt Post in September of 2017.
Credit goes to Rico Sennrich, who made the first move in this direction, and various existing BLEU implementations within Moses, which this code repeats.
