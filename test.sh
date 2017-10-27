#!/bin/bash

# Confirms that BLEU scores computed by sacreBLEU are the same as Moses' mteval-v13a.pl.
# Note that this doesn't work if you lowercase the data (remove -c to mteval-v13a.pl,
# or add "-lc" to sacreBLEU) because mteval-v13a.pl does not lowercase properly: it uses
# tr/[A-Z]/[a-z].

if [[ -z $MOSES ]]; then
    echo "Please define \$MOSES to point to your Moses installation."
    exit 1
fi

if [[ $(which sacrebleu > /dev/null) -ne 0 ]]; then
    echo "Please install sacreBLEU."
    exit 1
fi

[[ ! -d data ]] && mkdir data
cd data

if [[ ! -d wmt17-submitted-data ]]; then
    echo "Downloading and unpacking WMT'17 system submissions (46 MB)..."
    wget -q http://data.statmt.org/wmt17/translation-task/wmt17-submitted-data-v1.0.tgz
    tar xzf wmt17-submitted-data-v1.0.tgz
fi

for pair in cs-en de-en en-cs en-de en-fi en-lv en-ru en-tr fi-en lv-en ru-en tr-en zh-en; do
    source=$(echo $pair | cut -d- -f1)
    target=$(echo $pair | cut -d- -f2)    
    for sgm in wmt17-submitted-data/sgm/system-outputs/newstest2017/$pair/*.sgm; do
        sys=$(basename $sgm .sgm | perl -pe 's/newstest2017\.//')
        txt=$(dirname $sgm | perl -pe 's/sgm/txt/')/$(basename $sgm .sgm)
        src=wmt17-submitted-data/sgm/sources/newstest2017-$source$target-src.$source.sgm
        ref=wmt17-submitted-data/sgm/references/newstest2017-$source$target-ref.$target.sgm

        bleu1=$($MOSES/scripts/generic/mteval-v13a.pl -c -s $src -r $ref -t $sgm 2> /dev/null | grep "BLEU score" | cut -d' ' -f9)
        bleu1=$(echo "print($bleu1 * 100)" | python)
        bleu2=$(cat $txt | sacrebleu -t wmt17 -l $source-$target | cut -d' ' -f3)

        echo "$source-$target $sys mteval: $bleu1 sacreBLEU: $bleu2"
    done
done
