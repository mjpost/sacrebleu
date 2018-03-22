#!/bin/bash

# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# Confirms that BLEU scores computed by sacreBLEU are the same as Moses' mteval-v13a.pl.
# Note that this doesn't work if you lowercase the data (remove -c to mteval-v13a.pl,
# or add "-lc" to sacreBLEU) because mteval-v13a.pl does not lowercase properly: it uses
# tr/[A-Z]/[a-z], which doesn't cover uppercase letters with diacritics. Also, the
# Chinese preprocessing was applied to all zh sources, references, and system outputs
# (http://statmt.org/wmt17/tokenizeChinese.py), as was done for WMT17.

set -u

export SACREBLEU=$(pwd)/.sacrebleu

# TEST 1: download and process WMT17 data
[[ -d $SACREBLEU/wmt17 ]] && rm -f $SACREBLEU/wmt17/{en-*,*-en*}
./sacrebleu.py --echo src -t wmt17 -l cs-en > /dev/null

# Test loading via file instead of STDIN
./sacrebleu.py -t wmt17 -l en-de --echo ref > .wmt17.en-de.de.tmp
score=$(./sacrebleu.py -t wmt17 -l en-de -i .wmt17.en-de.de.tmp -b)
if [[ $score != '100.00' ]]; then
    echo "File test failed."
    exit 1
fi

[[ ! -d data ]] && mkdir data
cd data

if [[ ! -d wmt17-submitted-data ]]; then
   echo "Downloading and unpacking WMT'17 system submissions (46 MB)..."
   wget -q http://data.statmt.org/wmt17/translation-task/wmt17-submitted-data-v1.0.tgz
   tar xzf wmt17-submitted-data-v1.0.tgz
fi

# Test echoing of source, reference, and both
../sacrebleu.py -t wmt17/ms -l zh-en --echo src > .tmp.echo
diff .tmp.echo $SACREBLEU/wmt17/ms/zh-en.zh
if [[ $? -ne 0 ]]; then
    echo "Source echo failed."
    exit 1
fi
../sacrebleu.py -t wmt17/ms -l zh-en --echo ref | cut -f3 > .tmp.echo
diff .tmp.echo $SACREBLEU/wmt17/ms/zh-en.en.2
if [[ $? -ne 0 ]]; then
    echo "Source echo failed."
    exit 1
fi

export LC_ALL=C

# Pre-computed results from Moses' mteval-v13a.pl
declare -a MTEVAL=(23.15 25.12 27.45 30.95 28.98 34.56 30.1 33.12 33.17 28.09 32.48 32.97 17.67 26.04 35.12 20.51 19.95 20.2 16.18 20.22 16.55 20.12 14.18 14.69 13.69 13.79 12.64 13.06 15.25 22.8 22.67 26.33 26.08 26.6 27.11 26.56 16.61 26.02 26.71 21.24 20.8 26.65 15.45 18.16 28.3 26.69 17.15 20.28 9.33 20.72 15.87 11.47 1.05 15.96 14.46 13.72 22.04 15.51 13.64 16.76 18.31 16.19 17.01 10.15 18.63 14.35 16.63 10.84 17.91 20.14 20.79 21.05 16.93 17.48 17.99 22.3 25.37 25.32 23.83 32.05 10.08 27.11 28.41 29.79 9.98 16.03 10.37 9.81 11.62 19.93 13.93 16.43 30.52 25.93 34.87 23.92 31.05 29.02 32.07 18.58 21.31 36.28 35.84 19.96 16.13 6.13 21.13 27.62 22.48 14.31 16.23 12.42 16.79 15.72 22.02 20.0 21.92 18.98 33.88 33.95 34.71 31.47 31.18 36.94 15.9 34.51 30.77 12.42 17.91 13.52 17.54 18.05 12.64 22.18 25.62 16.38 20.08 22.32 18.98 25.78 18.51 16.47 20.76 26.38 15.94 21.28 20.48 24.51 33.23 11.39 15.5 25.7 26.0)
declare -i i=0
for pair in cs-en de-en en-cs en-de en-fi en-lv en-ru en-tr en-zh fi-en lv-en ru-en tr-en zh-en; do
    tokenizer=13a
    if [[ $pair == "en-zh" ]]; then
        tokenizer=zh
    fi
    source=$(echo $pair | cut -d- -f1)
    target=$(echo $pair | cut -d- -f2)
    for sgm in wmt17-submitted-data/sgm/system-outputs/newstest2017/$pair/*.sgm; do
        sys=$(basename $sgm .sgm | perl -pe 's/newstest2017\.//')
        txt=$(dirname $sgm | perl -pe 's/sgm/txt/')/$(basename $sgm .sgm)
        src=wmt17-submitted-data/sgm/sources/newstest2017-$source$target-src.$source.sgm
        ref=wmt17-submitted-data/sgm/references/newstest2017-$source$target-ref.$target.sgm

        # mteval=$($MOSES/scripts/generic/mteval-v13a.pl -c -s $src -r $ref -t $sgm 2> /dev/null | grep "BLEU score" | cut -d' ' -f9)
        # mteval=$(echo "print($bleu1 * 100)" | python)
        score=$(cat $txt | PYTHONIOENCODING=ascii ../sacrebleu.py -t wmt17 -l $source-$target --tok $tokenizer -b)

        echo "import sys; sys.exit(1 if abs($score-${MTEVAL[$i]}) > 0.04 else 0)" | python

        if [[ $? -eq 1 ]]; then
            echo "Failed test $pair/$sys ($score ${MTEVAL[$i]})"
            exit 1
        fi
#        echo "$source-$target $sys mteval: ${MTEVAL[$i]} sacreBLEU: $score mteval-v13a.pl"

        let i++
    done
done

echo "Tests passed."
exit 0
