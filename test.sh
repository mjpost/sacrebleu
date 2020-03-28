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

if [[ $(echo $BASH_VERSION | cut -d. -f1) -lt 4 ]]; then
    echo "This script requires BASH version 4 or above (since it uses hashes)."
    exit 1
fi

export SACREBLEU=$(pwd)/.sacrebleu
export PYTHONPATH="${PWD}"    # assuming PYTHONPATH=. as the default
CMD="python3 -m sacrebleu"

# Only run this test
limit_test=${1:-}

# TEST 1: download and process WMT17 data
[[ -d $SACREBLEU/wmt17 ]] && rm -f $SACREBLEU/wmt17/{en-*,*-en*}
${CMD} --echo src -t wmt17 -l cs-en > /dev/null

# Test concatenation of multiple test sets, --echo, -w, --origlang, --verbose
# and a pipeline with two sacrebleu processes
declare -A EXPECTED
EXPECTED["${CMD} -t wmt16,wmt17 -l en-fi --echo ref | ${CMD} -b -w 4 -t wmt16/B,wmt17/B -l en-fi"]=53.7432
EXPECTED["${CMD} -t wmt16,wmt17 -l en-fi --echo ref | ${CMD} -b -w 4 -t wmt16/B,wmt17/B -l en-fi --origlang=en"]=18.9054
EXPECTED["${CMD} -t wmt17 -l en-fi --echo ref | ${CMD} -b -t wmt17/B -l en-fi --detail"]="55.6
origlang=en                     : sentences=1502 BLEU= 21.4
origlang=fi                     : sentences=1500 BLEU=100.0"
EXPECTED["${CMD} -t wmt18,wmt19 -l en-de --echo=src | ${CMD} -t wmt18,wmt19 -l en-de -b --detail"]="3.6
origlang=de                     : sentences=1498 BLEU=  3.6
origlang=en                     : sentences=3497 BLEU=  3.5
origlang=en           country=EU: sentences= 265 BLEU=  2.5
origlang=en           country=GB: sentences= 913 BLEU=  3.1
origlang=en        country=OTHER: sentences= 801 BLEU=  2.5
origlang=en           country=US: sentences=1518 BLEU=  4.2
origlang=en      domain=business: sentences= 241 BLEU=  3.4
origlang=en         domain=crime: sentences= 570 BLEU=  3.6
origlang=en domain=entertainment: sentences= 322 BLEU=  5.1
origlang=en      domain=politics: sentences= 959 BLEU=  3.0
origlang=en       domain=scitech: sentences= 211 BLEU=  3.1
origlang=en         domain=sport: sentences= 534 BLEU=  3.6
origlang=en         domain=world: sentences= 660 BLEU=  3.1"

for command in "${!EXPECTED[@]}"; do
  echo Testing $command
  obtained=`eval $command`
  expected=${EXPECTED[$command]}
  if [[ $obtained != $expected ]]; then
      echo -e "\nFAILED:\n expected = $expected\n obtained = $obtained"
      exit 1
  fi
  echo PASS
done

# Test loading via file instead of STDIN
${CMD} -t wmt17 -l en-de --echo ref > .wmt17.en-de.de.tmp
score=$(${CMD} -t wmt17 -l en-de -i .wmt17.en-de.de.tmp -b)
if [[ $score != '100.0' ]]; then
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

if [[ ! -d en-ja-translation-example-master ]]; then
   echo "Downloading and unpacking English-Japanese test data..."
   wget -q https://github.com/MorinoseiMorizo/en-ja-translation-example/archive/master.zip
   unzip master.zip
fi

# Test echoing of source, reference, and both
${CMD} -t wmt17/ms -l zh-en --echo src > .tmp.echo
diff .tmp.echo $SACREBLEU/wmt17/ms/zh-en.zh
if [[ $? -ne 0 ]]; then
    echo "Source echo failed."
    exit 1
fi
${CMD} -t wmt17/ms -l zh-en --echo ref | cut -f3 > .tmp.echo
diff .tmp.echo $SACREBLEU/wmt17/ms/zh-en.en.2
if [[ $? -ne 0 ]]; then
    echo "Source echo failed."
    exit 1
fi

export LC_ALL=C

# Pre-computed results from Moses' mteval-v13a.pl
declare -A MTEVAL=( ["newstest2017.PJATK.4760.cs-en.sgm"]=23.15
                    ["newstest2017.online-A.0.cs-en.sgm"]=25.12
                    ["newstest2017.online-B.0.cs-en.sgm"]=27.45
                    ["newstest2017.uedin-nmt.4955.cs-en.sgm"]=30.95
                    ["newstest2017.C-3MA.4958.de-en.sgm"]=28.98
                    ["newstest2017.KIT.4951.de-en.sgm"]=34.56
                    ["newstest2017.LIUM-NMT.4733.de-en.sgm"]=30.1
                    ["newstest2017.RWTH-nmt-ensemble.4920.de-en.sgm"]=33.12
                    ["newstest2017.SYSTRAN.4846.de-en.sgm"]=33.17
                    ["newstest2017.TALP-UPC.4830.de-en.sgm"]=28.09
                    ["newstest2017.online-A.0.de-en.sgm"]=32.48
                    ["newstest2017.online-B.0.de-en.sgm"]=32.97
                    ["newstest2017.online-F.0.de-en.sgm"]=17.67
                    ["newstest2017.online-G.0.de-en.sgm"]=26.04
                    ["newstest2017.uedin-nmt.4723.de-en.sgm"]=35.12
                    ["newstest2017.CU-Chimera.4886.en-cs.sgm"]=20.51
                    ["newstest2017.LIUM-FNMT.4852.en-cs.sgm"]=19.95
                    ["newstest2017.LIUM-NMT.4947.en-cs.sgm"]=20.2
                    ["newstest2017.PJATK.4761.en-cs.sgm"]=16.18
                    ["newstest2017.limsi-factored-norm.4957.en-cs.sgm"]=20.22
                    ["newstest2017.online-A.0.en-cs.sgm"]=16.55
                    ["newstest2017.online-B.0.en-cs.sgm"]=20.12
                    ["newstest2017.tuning-task-afrl_4gb.sgm.0.en-cs.sgm"]=14.18
                    ["newstest2017.tuning-task-afrl_8gb.sgm.0.en-cs.sgm"]=14.69
                    ["newstest2017.tuning-task-baseline_4gb.sgm.0.en-cs.sgm"]=13.69
                    ["newstest2017.tuning-task-baseline_8gb.sgm.0.en-cs.sgm"]=13.79
                    ["newstest2017.tuning-task-denisov_4gb.sgm.0.en-cs.sgm"]=12.64
                    ["newstest2017.tuning-task-ufal_4gb.sgm.0.en-cs.sgm"]=13.06
                    ["newstest2017.tuning-task-ufal_8gb.sgm.0.en-cs.sgm"]=15.25
                    ["newstest2017.uedin-nmt.4956.en-cs.sgm"]=22.8
                    ["newstest2017.C-3MA.4959.en-de.sgm"]=22.67
                    ["newstest2017.FBK.4870.en-de.sgm"]=26.33
                    ["newstest2017.KIT.4950.en-de.sgm"]=26.08
                    ["newstest2017.LIUM-NMT.4900.en-de.sgm"]=26.6
                    ["newstest2017.LMU-nmt-reranked.4934.en-de.sgm"]=27.11
                    ["newstest2017.LMU-nmt-single.4893.en-de.sgm"]=26.56
                    ["newstest2017.PROMT-Rule-based.4735.en-de.sgm"]=16.61
                    ["newstest2017.RWTH-nmt-ensemble.4921.en-de.sgm"]=26.02
                    ["newstest2017.SYSTRAN.4847.en-de.sgm"]=26.71
                    ["newstest2017.TALP-UPC.4834.en-de.sgm"]=21.24
                    ["newstest2017.online-A.0.en-de.sgm"]=20.8
                    ["newstest2017.online-B.0.en-de.sgm"]=26.65
                    ["newstest2017.online-F.0.en-de.sgm"]=15.45
                    ["newstest2017.online-G.0.en-de.sgm"]=18.16
                    ["newstest2017.uedin-nmt.4722.en-de.sgm"]=28.3
                    ["newstest2017.xmu.4910.en-de.sgm"]=26.69
                    ["newstest2017.AaltoHnmtFlatcat.4798.en-fi.sgm"]=17.15
                    ["newstest2017.AaltoHnmtMultitask.4873.en-fi.sgm"]=20.28
                    ["newstest2017.HY-AH.4797.en-fi.sgm"]=9.33
                    ["newstest2017.HY-HNMT.4961.en-fi.sgm"]=20.72
                    ["newstest2017.HY-SMT.4882.en-fi.sgm"]=15.87
                    ["newstest2017.TALP-UPC.4939.en-fi.sgm"]=11.47
                    ["newstest2017.apertium-unconstrained.4769.en-fi.sgm"]=1.05
                    ["newstest2017.jhu-nmt-lattice-rescore.4903.en-fi.sgm"]=15.96
                    ["newstest2017.jhu-pbmt.4968.en-fi.sgm"]=14.46
                    ["newstest2017.online-A.0.en-fi.sgm"]=13.72
                    ["newstest2017.online-B.0.en-fi.sgm"]=22.04
                    ["newstest2017.online-G.0.en-fi.sgm"]=15.51
                    ["newstest2017.C-3MA.5069.en-lv.sgm"]=13.64
                    ["newstest2017.HY-HNMT.5066.en-lv.sgm"]=16.76
                    ["newstest2017.KIT.5062.en-lv.sgm"]=18.31
                    ["newstest2017.LIUM-FNMT.5043.en-lv.sgm"]=16.19
                    ["newstest2017.LIUM-NMT.5042.en-lv.sgm"]=17.01
                    ["newstest2017.PJATK.4744.en-lv.sgm"]=10.15
                    ["newstest2017.QT21-System-Combination.5063.en-lv.sgm"]=18.63
                    ["newstest2017.jhu-pbmt.4969.en-lv.sgm"]=14.35
                    ["newstest2017.limsi-factored-norm.5041.en-lv.sgm"]=16.63
                    ["newstest2017.online-A.0.en-lv.sgm"]=10.84
                    ["newstest2017.online-B.0.en-lv.sgm"]=17.91
                    ["newstest2017.tilde-c-nmt-smt-hybrid.5049.en-lv.sgm"]=20.14
                    ["newstest2017.tilde-nc-nmt-smt-hybrid.5047.en-lv.sgm"]=20.79
                    ["newstest2017.tilde-nc-smt.5044.en-lv.sgm"]=21.05
                    ["newstest2017.uedin-nmt.5016.en-lv.sgm"]=16.93
                    ["newstest2017.usfd-consensus-kit.5078.en-lv.sgm"]=17.48
                    ["newstest2017.usfd-consensus-qt21.5077.en-lv.sgm"]=17.99
                    ["newstest2017.PROMT-Rule-based.4736.en-ru.sgm"]=22.3
                    ["newstest2017.afrl-mitll-backtrans.4907.en-ru.sgm"]=25.37
                    ["newstest2017.jhu-pbmt.4986.en-ru.sgm"]=25.32
                    ["newstest2017.online-A.0.en-ru.sgm"]=23.83
                    ["newstest2017.online-B.0.en-ru.sgm"]=32.05
                    ["newstest2017.online-F.0.en-ru.sgm"]=10.08
                    ["newstest2017.online-G.0.en-ru.sgm"]=27.11
                    ["newstest2017.online-H.0.en-ru.sgm"]=28.41
                    ["newstest2017.uedin-nmt.4756.en-ru.sgm"]=29.79
                    ["newstest2017.JAIST.4858.en-tr.sgm"]=9.98
                    ["newstest2017.LIUM-NMT.4953.en-tr.sgm"]=16.03
                    ["newstest2017.jhu-nmt-lattice-rescore.4904.en-tr.sgm"]=10.37
                    ["newstest2017.jhu-pbmt.4970.en-tr.sgm"]=9.81
                    ["newstest2017.online-A.0.en-tr.sgm"]=11.62
                    ["newstest2017.online-B.0.en-tr.sgm"]=19.93
                    ["newstest2017.online-G.0.en-tr.sgm"]=13.93
                    ["newstest2017.uedin-nmt.4932.en-tr.sgm"]=16.43
                    ["newstest2017.CASICT-DCU-NMT.5157.en-zh.sgm"]=30.52
                    ["newstest2017.Oregon-State-University-S.5174.en-zh.sgm"]=25.93
                    ["newstest2017.SogouKnowing-nmt.5131.en-zh.sgm"]=34.87
                    ["newstest2017.UU-HNMT.5134.en-zh.sgm"]=23.92
                    ["newstest2017.jhu-nmt.5153.en-zh.sgm"]=31.05
                    ["newstest2017.online-A.0.en-zh.sgm"]=29.02
                    ["newstest2017.online-B.0.en-zh.sgm"]=32.07
                    ["newstest2017.online-F.0.en-zh.sgm"]=18.58
                    ["newstest2017.online-G.0.en-zh.sgm"]=21.31
                    ["newstest2017.uedin-nmt.5111.en-zh.sgm"]=36.28
                    ["newstest2017.xmunmt.5165.en-zh.sgm"]=35.84
                    ["newstest2017.Hunter-MT.4925.fi-en.sgm"]=19.96
                    ["newstest2017.TALP-UPC.4937.fi-en.sgm"]=16.13
                    ["newstest2017.apertium-unconstrained.4793.fi-en.sgm"]=6.13
                    ["newstest2017.online-A.0.fi-en.sgm"]=21.13
                    ["newstest2017.online-B.0.fi-en.sgm"]=27.62
                    ["newstest2017.online-G.0.fi-en.sgm"]=22.48
                    ["newstest2017.C-3MA.5067.lv-en.sgm"]=14.31
                    ["newstest2017.Hunter-MT.5092.lv-en.sgm"]=16.23
                    ["newstest2017.PJATK.4740.lv-en.sgm"]=12.42
                    ["newstest2017.jhu-pbmt.4980.lv-en.sgm"]=16.79
                    ["newstest2017.online-A.0.lv-en.sgm"]=15.72
                    ["newstest2017.online-B.0.lv-en.sgm"]=22.02
                    ["newstest2017.tilde-c-nmt-smt-hybrid.5051.lv-en.sgm"]=20.0
                    ["newstest2017.tilde-nc-nmt-smt-hybrid.5050.lv-en.sgm"]=21.92
                    ["newstest2017.uedin-nmt.5017.lv-en.sgm"]=18.98
                    ["newstest2017.NRC.4855.ru-en.sgm"]=33.88
                    ["newstest2017.afrl-mitll-opennmt.4896.ru-en.sgm"]=33.95
                    ["newstest2017.afrl-mitll-syscomb.4905.ru-en.sgm"]=34.71
                    ["newstest2017.jhu-pbmt.4978.ru-en.sgm"]=31.47
                    ["newstest2017.online-A.0.ru-en.sgm"]=31.18
                    ["newstest2017.online-B.0.ru-en.sgm"]=36.94
                    ["newstest2017.online-F.0.ru-en.sgm"]=15.9
                    ["newstest2017.online-G.0.ru-en.sgm"]=34.51
                    ["newstest2017.uedin-nmt.4890.ru-en.sgm"]=30.77
                    ["newstest2017.JAIST.4859.tr-en.sgm"]=12.42
                    ["newstest2017.LIUM-NMT.4888.tr-en.sgm"]=17.91
                    ["newstest2017.PROMT-SMT.4737.tr-en.sgm"]=13.52
                    ["newstest2017.afrl-mitll-m2w-nr1.4901.tr-en.sgm"]=17.54
                    ["newstest2017.afrl-mitll-syscomb.4902.tr-en.sgm"]=18.05
                    ["newstest2017.jhu-pbmt.4972.tr-en.sgm"]=12.64
                    ["newstest2017.online-A.0.tr-en.sgm"]=22.18
                    ["newstest2017.online-B.0.tr-en.sgm"]=25.62
                    ["newstest2017.online-G.0.tr-en.sgm"]=16.38
                    ["newstest2017.uedin-nmt.4931.tr-en.sgm"]=20.08
                    ["newstest2017.CASICT-DCU-NMT.5144.zh-en.sgm"]=22.32
                    ["newstest2017.NMT-Model-Average-Multi-Cards.5099.zh-en.sgm"]=18.98
                    ["newstest2017.NRC.5172.zh-en.sgm"]=25.78
                    ["newstest2017.Oregon-State-University-S.5173.zh-en.sgm"]=18.51
                    ["newstest2017.PROMT-SMT.5125.zh-en.sgm"]=16.47
                    ["newstest2017.ROCMT.5183.zh-en.sgm"]=20.76
                    ["newstest2017.SogouKnowing-nmt.5171.zh-en.sgm"]=26.38
                    ["newstest2017.UU-HNMT.5162.zh-en.sgm"]=15.94
                    ["newstest2017.afrl-mitll-opennmt.5109.zh-en.sgm"]=21.28
                    ["newstest2017.jhu-nmt.5151.zh-en.sgm"]=20.48
                    ["newstest2017.online-A.0.zh-en.sgm"]=24.51
                    ["newstest2017.online-B.0.zh-en.sgm"]=33.23
                    ["newstest2017.online-F.0.zh-en.sgm"]=11.39
                    ["newstest2017.online-G.0.zh-en.sgm"]=15.5
                    ["newstest2017.uedin-nmt.5112.zh-en.sgm"]=25.7
                    ["newstest2017.xmunmt.5160.zh-en.sgm"]=26.0
                    ["kyoto-test"]=14.48
                  )

declare -i i=0
for pair in cs-en de-en en-cs en-de en-fi en-lv en-ru en-tr en-zh fi-en lv-en ru-en tr-en zh-en; do
    source=$(echo $pair | cut -d- -f1)
    target=$(echo $pair | cut -d- -f2)
    for sgm in wmt17-submitted-data/sgm/system-outputs/newstest2017/$pair/*.sgm; do
        name=$(basename $sgm)

        if [[ ! -z $limit_test && $limit_test != $name ]]; then continue; fi

        sys=$(basename $sgm .sgm | perl -pe 's/newstest2017\.//')
        txt=$(dirname $sgm | perl -pe 's/sgm/txt/')/$(basename $sgm .sgm)
        src=wmt17-submitted-data/sgm/sources/newstest2017-$source$target-src.$source.sgm
        ref=wmt17-submitted-data/sgm/references/newstest2017-$source$target-ref.$target.sgm

        # mteval=$($MOSES/scripts/generic/mteval-v13a.pl -c -s $src -r $ref -t $sgm 2> /dev/null | grep "BLEU score" | cut -d' ' -f9)
        # mteval=$(echo "print($bleu1 * 100)" | python)
        score=$(cat $txt | ${CMD} -w 2 -t wmt17 -l $source-$target -b)

        echo "import sys; sys.exit(1 if abs($score-${MTEVAL[$name]}) > 0.01 else 0)" | python

        if [[ $? -eq 1 ]]; then
            echo "FAILED test $pair/$sys (wanted ${MTEVAL[$name]} got $score)"
            exit 1
        fi
        echo "Passed $source-$target $sys mteval-v13a.pl: ${MTEVAL[$name]} sacreBLEU: $score"

        let i++
    done
done

for pair in en-ja; do
    source=$(echo $pair | cut -d- -f1)
    target=$(echo $pair | cut -d- -f2)
    for txt in en-ja-translation-example-master/*.hyp.$target; do
        name=$(basename $txt .hyp.$target)

        if [[ ! -z $limit_test && $limit_test != $name ]]; then continue; fi

        sys=$(basename $txt .hyp.$target)
        ref=$(dirname $txt)/$(basename $txt .hyp.$target).ref.$target
        score=$(cat $txt | ${CMD}  -w 2 -l $source-$target -b $ref)

        echo "import sys; sys.exit(1 if abs($score-${MTEVAL[$name]}) > 0.01 else 0)" | python

        if [[ $? -eq 1 ]]; then
            echo "FAILED test $pair/$sys (wanted ${MTEVAL[$name]} got $score)"
            exit 1
        fi
        echo "Passed $source-$target $sys mteval-v13a.pl: ${MTEVAL[$name]} sacreBLEU: $score"

        let i++
    done
done

score1=$( echo "Hello! How are you doing today?" | ${CMD} -w 2 -b <(printf "Hello! How are you \r doing today?") )
score2=$( echo "Hello! How are you doing today?" | ${CMD} -w 2 -b <(echo "Hello! How are you doing today?") )
if [[ $score1 != $score2 ]]; then
  echo "Control character in reference test failed"
  exit 1
fi
let i++
echo "Passed control character in reference test"

echo "Passed $i tests."
exit 0
