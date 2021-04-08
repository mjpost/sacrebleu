#!/usr/bin/env bash

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

SKIP_CHRF=${SKIP_CHRF:-}
SKIP_TER=${SKIP_TER:-}
SKIP_MECAB=${SKIP_MECAB:-}

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
echo "Testing loading via file instead of STDIN"
${CMD} -t wmt17 -l en-de --echo ref > .wmt17.en-de.de.tmp
score=$(${CMD} -t wmt17 -l en-de -i .wmt17.en-de.de.tmp -b)
rm .wmt17.en-de.de.tmp
if [[ $score != '100.0' ]]; then
    echo "File test failed."
    exit 1
fi
echo PASS

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

declare -i i=0

echo '-----------------------'
echo 'Control character tests'
echo '-----------------------'
score1=$( echo "Hello! How are you doing today?" | ${CMD} -w 2 -b <(printf "Hello! How are you \r doing today?") )
score2=$( echo "Hello! How are you doing today?" | ${CMD} -w 2 -b <(echo "Hello! How are you doing today?") )
if [[ $score1 != $score2 ]]; then
  echo "Control character in reference test failed"
  exit 1
fi
let i++
echo "Passed control character in reference test"

#####################################################################
# Tests for single-ref BLEU, multi-ref BLEU, signature and tokenizers
#####################################################################
path="wmt17-submitted-data/txt/system-outputs/newstest2017/cs-en"
ref1="${path}/newstest2017.online-A.0.cs-en"
ref2="${path}/newstest2017.online-B.0.cs-en"
sys="${path}/newstest2017.PJATK.4760.cs-en"

echo '---------------------'
echo 'BLEU regression tests'
echo '---------------------'
unset EXPECTED
declare -A EXPECTED

# Single ref, tokenizer variants, lowercase
EXPECTED["${CMD} -w 4 -b -l cs-en -i $sys $ref1"]=36.8799
EXPECTED["${CMD} -lc -w 4 -b -l cs-en -i $sys $ref1"]=38.1492
EXPECTED["${CMD} --tokenize 13a  -w 4 -b -l cs-en -i $sys $ref1"]=36.8799
EXPECTED["${CMD} --tokenize none -w 4 -b -l cs-en -i $sys $ref1"]=34.0638
EXPECTED["${CMD} --tokenize intl -w 4 -b -l cs-en -i $sys $ref1"]=37.3859
# multiple REF files
EXPECTED["${CMD} -w 4 -b -l cs-en -i $sys $ref1 $ref2"]=44.6732
# multiple REFs with tab-delimited stream
EXPECTED["${CMD} -w 4 -b -l cs-en -i $sys --num-refs 2 <(paste $ref1 $ref2)"]=44.6732
# Check signature correctness for multi-reference
# separate files
EXPECTED["${CMD} -l cs-en -i $sys $ref1 $ref2 | perl -pe 's/.*numrefs\.([0-9]).*/\1/'"]=2
# tab delimited stream
EXPECTED["${CMD} -l cs-en -i $sys --num-refs 2 <(paste $ref1 $ref2) | perl -pe 's/.*numrefs\.([0-9]).*/\1/'"]=2


# Run the tests
for command in "${!EXPECTED[@]}"; do
  echo Testing $command
  obtained=`eval $command`
  expected=${EXPECTED[$command]}
  if [[ $obtained != $expected ]]; then
      echo -e "\nFAILED:\n expected = $expected\n obtained = $obtained"
      exit 1
  fi
  echo PASS
  let i++
done

#######################################################
# Pre-computed chrF scores from official implementation
# Cmd: chrF++.py -H hyp -R ref -nw 0
#######################################################
declare -A CHRF=( ["newstest2017.PJATK.4760.cs-en.sgm"]=52.5947
                  ["newstest2017.online-A.0.cs-en.sgm"]=53.3856
                  ["newstest2017.online-B.0.cs-en.sgm"]=54.4608
                  ["newstest2017.uedin-nmt.4955.cs-en.sgm"]=56.8490
                  ["newstest2017.C-3MA.4958.de-en.sgm"]=54.9500
                  ["newstest2017.KIT.4951.de-en.sgm"]=59.5876
                  ["newstest2017.LIUM-NMT.4733.de-en.sgm"]=56.1531
                  ["newstest2017.RWTH-nmt-ensemble.4920.de-en.sgm"]=58.8482
                  ["newstest2017.SYSTRAN.4846.de-en.sgm"]=58.6623
                  ["newstest2017.TALP-UPC.4830.de-en.sgm"]=55.6962
                  ["newstest2017.online-A.0.de-en.sgm"]=59.1026
                  ["newstest2017.online-B.0.de-en.sgm"]=59.0564
                  ["newstest2017.online-F.0.de-en.sgm"]=50.2126
                  ["newstest2017.online-G.0.de-en.sgm"]=55.6530
                  ["newstest2017.uedin-nmt.4723.de-en.sgm"]=60.1464
                  ["newstest2017.CU-Chimera.4886.en-cs.sgm"]=48.3370
                  ["newstest2017.LIUM-FNMT.4852.en-cs.sgm"]=48.4708
                  ["newstest2017.LIUM-NMT.4947.en-cs.sgm"]=48.4079
                  ["newstest2017.PJATK.4761.en-cs.sgm"]=43.0152
                  ["newstest2017.limsi-factored-norm.4957.en-cs.sgm"]=48.7015
                  ["newstest2017.online-A.0.en-cs.sgm"]=45.9326
                  ["newstest2017.online-B.0.en-cs.sgm"]=48.4691
                  ["newstest2017.tuning-task-afrl_4gb.sgm.0.en-cs.sgm"]=40.8498
                  ["newstest2017.tuning-task-afrl_8gb.sgm.0.en-cs.sgm"]=41.3727
                  ["newstest2017.tuning-task-baseline_4gb.sgm.0.en-cs.sgm"]=40.4781
                  ["newstest2017.tuning-task-baseline_8gb.sgm.0.en-cs.sgm"]=40.5823
                  ["newstest2017.tuning-task-denisov_4gb.sgm.0.en-cs.sgm"]=39.9792
                  ["newstest2017.tuning-task-ufal_4gb.sgm.0.en-cs.sgm"]=39.4850
                  ["newstest2017.tuning-task-ufal_8gb.sgm.0.en-cs.sgm"]=42.4445
                  ["newstest2017.uedin-nmt.4956.en-cs.sgm"]=50.5857
                  ["newstest2017.C-3MA.4959.en-de.sgm"]=51.9533
                  ["newstest2017.FBK.4870.en-de.sgm"]=54.7152
                  ["newstest2017.KIT.4950.en-de.sgm"]=55.7629
                  ["newstest2017.LIUM-NMT.4900.en-de.sgm"]=55.9284
                  ["newstest2017.LMU-nmt-reranked.4934.en-de.sgm"]=56.3908
                  ["newstest2017.LMU-nmt-single.4893.en-de.sgm"]=55.9216
                  ["newstest2017.PROMT-Rule-based.4735.en-de.sgm"]=50.3511
                  ["newstest2017.RWTH-nmt-ensemble.4921.en-de.sgm"]=55.6116
                  ["newstest2017.SYSTRAN.4847.en-de.sgm"]=55.5758
                  ["newstest2017.TALP-UPC.4834.en-de.sgm"]=51.6860
                  ["newstest2017.online-A.0.en-de.sgm"]=52.0023
                  ["newstest2017.online-B.0.en-de.sgm"]=56.2633
                  ["newstest2017.online-F.0.en-de.sgm"]=49.2588
                  ["newstest2017.online-G.0.en-de.sgm"]=51.5871
                  ["newstest2017.uedin-nmt.4722.en-de.sgm"]=57.7227
                  ["newstest2017.xmu.4910.en-de.sgm"]=55.9642
                  ["newstest2017.AaltoHnmtFlatcat.4798.en-fi.sgm"]=50.5981
                  ["newstest2017.AaltoHnmtMultitask.4873.en-fi.sgm"]=52.4618
                  ["newstest2017.HY-AH.4797.en-fi.sgm"]=46.9995
                  ["newstest2017.HY-HNMT.4961.en-fi.sgm"]=54.9460
                  ["newstest2017.HY-SMT.4882.en-fi.sgm"]=51.2609
                  ["newstest2017.TALP-UPC.4939.en-fi.sgm"]=44.8177
                  ["newstest2017.apertium-unconstrained.4769.en-fi.sgm"]=21.7725
                  ["newstest2017.jhu-nmt-lattice-rescore.4903.en-fi.sgm"]=51.3314
                  ["newstest2017.jhu-pbmt.4968.en-fi.sgm"]=49.7043
                  ["newstest2017.online-A.0.en-fi.sgm"]=49.5458
                  ["newstest2017.online-B.0.en-fi.sgm"]=56.1894
                  ["newstest2017.online-G.0.en-fi.sgm"]=51.8957
                  ["newstest2017.C-3MA.5069.en-lv.sgm"]=43.8029
                  ["newstest2017.HY-HNMT.5066.en-lv.sgm"]=46.3223
                  ["newstest2017.KIT.5062.en-lv.sgm"]=51.1055
                  ["newstest2017.LIUM-FNMT.5043.en-lv.sgm"]=47.9871
                  ["newstest2017.LIUM-NMT.5042.en-lv.sgm"]=48.1380
                  ["newstest2017.PJATK.4744.en-lv.sgm"]=35.9152
                  ["newstest2017.QT21-System-Combination.5063.en-lv.sgm"]=50.6553
                  ["newstest2017.jhu-pbmt.4969.en-lv.sgm"]=46.9511
                  ["newstest2017.limsi-factored-norm.5041.en-lv.sgm"]=49.3634
                  ["newstest2017.online-A.0.en-lv.sgm"]=45.2101
                  ["newstest2017.online-B.0.en-lv.sgm"]=50.1384
                  ["newstest2017.tilde-c-nmt-smt-hybrid.5049.en-lv.sgm"]=51.6770
                  ["newstest2017.tilde-nc-nmt-smt-hybrid.5047.en-lv.sgm"]=52.7970
                  ["newstest2017.tilde-nc-smt.5044.en-lv.sgm"]=51.5999
                  ["newstest2017.uedin-nmt.5016.en-lv.sgm"]=49.2607
                  ["newstest2017.usfd-consensus-kit.5078.en-lv.sgm"]=50.7400
                  ["newstest2017.usfd-consensus-qt21.5077.en-lv.sgm"]=51.0538
                  ["newstest2017.PROMT-Rule-based.4736.en-ru.sgm"]=53.2902
                  ["newstest2017.afrl-mitll-backtrans.4907.en-ru.sgm"]=52.2807
                  ["newstest2017.jhu-pbmt.4986.en-ru.sgm"]=54.9569
                  ["newstest2017.online-A.0.en-ru.sgm"]=53.4180
                  ["newstest2017.online-B.0.en-ru.sgm"]=60.4059
                  ["newstest2017.online-F.0.en-ru.sgm"]=42.0595
                  ["newstest2017.online-G.0.en-ru.sgm"]=56.5493
                  ["newstest2017.online-H.0.en-ru.sgm"]=56.8716
                  ["newstest2017.uedin-nmt.4756.en-ru.sgm"]=56.6076
                  ["newstest2017.JAIST.4858.en-tr.sgm"]=42.1117
                  ["newstest2017.LIUM-NMT.4953.en-tr.sgm"]=47.5881
                  ["newstest2017.jhu-nmt-lattice-rescore.4904.en-tr.sgm"]=42.5309
                  ["newstest2017.jhu-pbmt.4970.en-tr.sgm"]=42.1480
                  ["newstest2017.online-A.0.en-tr.sgm"]=47.4192
                  ["newstest2017.online-B.0.en-tr.sgm"]=54.1855
                  ["newstest2017.online-G.0.en-tr.sgm"]=48.7404
                  ["newstest2017.uedin-nmt.4932.en-tr.sgm"]=50.3093
                  ["newstest2017.CASICT-DCU-NMT.5157.en-zh.sgm"]=27.0468
                  ["newstest2017.Oregon-State-University-S.5174.en-zh.sgm"]=24.5325
                  ["newstest2017.SogouKnowing-nmt.5131.en-zh.sgm"]=31.3259
                  ["newstest2017.UU-HNMT.5134.en-zh.sgm"]=22.6901
                  ["newstest2017.jhu-nmt.5153.en-zh.sgm"]=27.9123
                  ["newstest2017.online-A.0.en-zh.sgm"]=25.6325
                  ["newstest2017.online-B.0.en-zh.sgm"]=29.2984
                  ["newstest2017.online-F.0.en-zh.sgm"]=18.7403
                  ["newstest2017.online-G.0.en-zh.sgm"]=20.6007
                  ["newstest2017.uedin-nmt.5111.en-zh.sgm"]=31.8748
                  ["newstest2017.xmunmt.5165.en-zh.sgm"]=31.7770
                  ["newstest2017.Hunter-MT.4925.fi-en.sgm"]=47.9929
                  ["newstest2017.TALP-UPC.4937.fi-en.sgm"]=45.7795
                  ["newstest2017.apertium-unconstrained.4793.fi-en.sgm"]=38.6486
                  ["newstest2017.online-A.0.fi-en.sgm"]=51.9119
                  ["newstest2017.online-B.0.fi-en.sgm"]=55.7417
                  ["newstest2017.online-G.0.fi-en.sgm"]=53.8541
                  ["newstest2017.C-3MA.5067.lv-en.sgm"]=43.3150
                  ["newstest2017.Hunter-MT.5092.lv-en.sgm"]=46.1868
                  ["newstest2017.PJATK.4740.lv-en.sgm"]=39.3033
                  ["newstest2017.jhu-pbmt.4980.lv-en.sgm"]=46.7783
                  ["newstest2017.online-A.0.lv-en.sgm"]=47.1552
                  ["newstest2017.online-B.0.lv-en.sgm"]=51.4714
                  ["newstest2017.tilde-c-nmt-smt-hybrid.5051.lv-en.sgm"]=49.1392
                  ["newstest2017.tilde-nc-nmt-smt-hybrid.5050.lv-en.sgm"]=51.5697
                  ["newstest2017.uedin-nmt.5017.lv-en.sgm"]=48.0781
                  ["newstest2017.NRC.4855.ru-en.sgm"]=60.1860
                  ["newstest2017.afrl-mitll-opennmt.4896.ru-en.sgm"]=59.4356
                  ["newstest2017.afrl-mitll-syscomb.4905.ru-en.sgm"]=59.7636
                  ["newstest2017.jhu-pbmt.4978.ru-en.sgm"]=58.1248
                  ["newstest2017.online-A.0.ru-en.sgm"]=57.9992
                  ["newstest2017.online-B.0.ru-en.sgm"]=63.0622
                  ["newstest2017.online-F.0.ru-en.sgm"]=49.5420
                  ["newstest2017.online-G.0.ru-en.sgm"]=61.8913
                  ["newstest2017.uedin-nmt.4890.ru-en.sgm"]=57.4335
                  ["newstest2017.JAIST.4859.tr-en.sgm"]=43.1983
                  ["newstest2017.LIUM-NMT.4888.tr-en.sgm"]=45.3857
                  ["newstest2017.PROMT-SMT.4737.tr-en.sgm"]=46.1464
                  ["newstest2017.afrl-mitll-m2w-nr1.4901.tr-en.sgm"]=45.7267
                  ["newstest2017.afrl-mitll-syscomb.4902.tr-en.sgm"]=46.1653
                  ["newstest2017.jhu-pbmt.4972.tr-en.sgm"]=43.2728
                  ["newstest2017.online-A.0.tr-en.sgm"]=52.1165
                  ["newstest2017.online-B.0.tr-en.sgm"]=54.1508
                  ["newstest2017.online-G.0.tr-en.sgm"]=49.4456
                  ["newstest2017.uedin-nmt.4931.tr-en.sgm"]=47.8457
                  ["newstest2017.CASICT-DCU-NMT.5144.zh-en.sgm"]=49.7426
                  ["newstest2017.NMT-Model-Average-Multi-Cards.5099.zh-en.sgm"]=47.3694
                  ["newstest2017.NRC.5172.zh-en.sgm"]=53.6810
                  ["newstest2017.Oregon-State-University-S.5173.zh-en.sgm"]=47.6272
                  ["newstest2017.PROMT-SMT.5125.zh-en.sgm"]=48.3674
                  ["newstest2017.ROCMT.5183.zh-en.sgm"]=50.0904
                  ["newstest2017.SogouKnowing-nmt.5171.zh-en.sgm"]=55.0223
                  ["newstest2017.UU-HNMT.5162.zh-en.sgm"]=45.2487
                  ["newstest2017.afrl-mitll-opennmt.5109.zh-en.sgm"]=50.3686
                  ["newstest2017.jhu-nmt.5151.zh-en.sgm"]=49.3613
                  ["newstest2017.online-A.0.zh-en.sgm"]=53.8268
                  ["newstest2017.online-B.0.zh-en.sgm"]=59.2377
                  ["newstest2017.online-F.0.zh-en.sgm"]=45.6546
                  ["newstest2017.online-G.0.zh-en.sgm"]=49.9084
                  ["newstest2017.uedin-nmt.5112.zh-en.sgm"]=53.5398
                  ["newstest2017.xmunmt.5160.zh-en.sgm"]=54.3314
                  )

if [ -z $SKIP_CHRF ]; then
  echo "-------------------"
  echo "Starting chrF tests"
  echo "-------------------"
  # Test only for different target languages as there is no tokenization
  # issue involved in chrF
  for pair in cs-en en-cs en-de en-fi en-lv en-ru en-tr en-zh; do
      source=$(echo $pair | cut -d- -f1)
      target=$(echo $pair | cut -d- -f2)
      for sgm in wmt17-submitted-data/sgm/system-outputs/newstest2017/$pair/*.sgm; do
          name=$(basename $sgm)

          if [[ ! -z $limit_test && $limit_test != $name ]]; then continue; fi

          sys=$(basename $sgm .sgm | perl -pe 's/newstest2017\.//')
          txt=$(dirname $sgm | perl -pe 's/sgm/txt/')/$(basename $sgm .sgm)
          src=wmt17-submitted-data/sgm/sources/newstest2017-$source$target-src.$source.sgm
          ref=wmt17-submitted-data/sgm/references/newstest2017-$source$target-ref.$target.sgm

          score=$(cat $txt | ${CMD} -w 4 -t wmt17 -l $source-$target -b --metrics chrf)

          # rescale to 0-1
          expected_score=`echo "print('{:.4f}'.format(${CHRF[$name]} / 100.0))" | python`

          echo "import sys; sys.exit(1 if abs(${score}-${expected_score}) > 0.01 else 0)" | python

          if [[ $? -eq 1 ]]; then
              echo "FAILED test $pair/$sys (wanted $expected_score got $score)"
              exit 1
          fi
          echo "Passed $source-$target $sys chrF++.py: $expected_score sacreCHRF: $score"

          let i++
      done
  done
fi

################################################################
# Pre-computed results from Moses' mteval-v13a.pl for BLEU tests
################################################################
echo "-------------------"
echo "Starting BLEU tests"
echo "-------------------"
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

#######################################################
# Pre-computed TER scores from official implementation
# Cmd: java -jar tercom.7.25.jar -r REF -h HYP | grep '^Total TER' | awk '{ printf "%.3f\n", $3 }'
#######################################################
declare -A TER=( ["newstest2017.online-A.0.cs-en.sgm"]=0.637
                 ["newstest2017.online-B.0.cs-en.sgm"]=0.605
                 ["newstest2017.PJATK.4760.cs-en.sgm"]=0.664
                 ["newstest2017.uedin-nmt.4955.cs-en.sgm"]=0.584
                 ["newstest2017.CU-Chimera.4886.en-cs.sgm"]=0.696
                 ["newstest2017.limsi-factored-norm.4957.en-cs.sgm"]=0.696
                 ["newstest2017.LIUM-FNMT.4852.en-cs.sgm"]=0.699
                 ["newstest2017.LIUM-NMT.4947.en-cs.sgm"]=0.701
                 ["newstest2017.online-A.0.en-cs.sgm"]=0.743
                 ["newstest2017.online-B.0.en-cs.sgm"]=0.703
                 ["newstest2017.PJATK.4761.en-cs.sgm"]=0.757
                 ["newstest2017.uedin-nmt.4956.en-cs.sgm"]=0.667
                 ["newstest2017.C-3MA.4959.en-de.sgm"]=0.669
                 ["newstest2017.FBK.4870.en-de.sgm"]=0.636
                 ["newstest2017.KIT.4950.en-de.sgm"]=0.641
                 ["newstest2017.LIUM-NMT.4900.en-de.sgm"]=0.633
                 ["newstest2017.LMU-nmt-reranked.4934.en-de.sgm"]=0.618
                 ["newstest2017.LMU-nmt-single.4893.en-de.sgm"]=0.626
                 ["newstest2017.online-A.0.en-de.sgm"]=0.690
                 ["newstest2017.online-B.0.en-de.sgm"]=0.632
                 ["newstest2017.online-F.0.en-de.sgm"]=0.756
                 ["newstest2017.online-G.0.en-de.sgm"]=0.733
                 ["newstest2017.PROMT-Rule-based.4735.en-de.sgm"]=0.752
                 ["newstest2017.RWTH-nmt-ensemble.4921.en-de.sgm"]=0.628
                 ["newstest2017.SYSTRAN.4847.en-de.sgm"]=0.611
                 ["newstest2017.TALP-UPC.4834.en-de.sgm"]=0.685
                 ["newstest2017.uedin-nmt.4722.en-de.sgm"]=0.612
                 ["newstest2017.xmu.4910.en-de.sgm"]=0.622
                 ["newstest2017.AaltoHnmtFlatcat.4798.en-fi.sgm"]=0.750
                 ["newstest2017.AaltoHnmtMultitask.4873.en-fi.sgm"]=0.673
                 ["newstest2017.apertium-unconstrained.4769.en-fi.sgm"]=1.246
                 ["newstest2017.HY-AH.4797.en-fi.sgm"]=0.857
                 ["newstest2017.HY-HNMT.4961.en-fi.sgm"]=0.676
                 ["newstest2017.HY-SMT.4882.en-fi.sgm"]=0.754
                 ["newstest2017.jhu-nmt-lattice-rescore.4903.en-fi.sgm"]=0.758
                 ["newstest2017.jhu-pbmt.4968.en-fi.sgm"]=0.778
                 ["newstest2017.online-A.0.en-fi.sgm"]=0.761
                 ["newstest2017.online-B.0.en-fi.sgm"]=0.665
                 ["newstest2017.online-G.0.en-fi.sgm"]=0.767
                 ["newstest2017.TALP-UPC.4939.en-fi.sgm"]=0.806
                 ["newstest2017.C-3MA.5069.en-lv.sgm"]=0.843
                 ["newstest2017.HY-HNMT.5066.en-lv.sgm"]=0.786
                 ["newstest2017.jhu-pbmt.4969.en-lv.sgm"]=0.815
                 ["newstest2017.KIT.5062.en-lv.sgm"]=0.773
                 ["newstest2017.limsi-factored-norm.5041.en-lv.sgm"]=0.780
                 ["newstest2017.LIUM-FNMT.5043.en-lv.sgm"]=0.780
                 ["newstest2017.LIUM-NMT.5042.en-lv.sgm"]=0.769
                 ["newstest2017.online-A.0.en-lv.sgm"]=0.846
                 ["newstest2017.online-B.0.en-lv.sgm"]=0.754
                 ["newstest2017.PJATK.4744.en-lv.sgm"]=0.868
                 ["newstest2017.tilde-c-nmt-smt-hybrid.5049.en-lv.sgm"]=0.731
                 ["newstest2017.tilde-nc-nmt-smt-hybrid.5047.en-lv.sgm"]=0.721
                 ["newstest2017.tilde-nc-smt.5044.en-lv.sgm"]=0.747
                 ["newstest2017.uedin-nmt.5016.en-lv.sgm"]=0.772
                 ["newstest2017.usfd-consensus-kit.5078.en-lv.sgm"]=0.950
                 ["newstest2017.usfd-consensus-qt21.5077.en-lv.sgm"]=0.946
                 ["newstest2017.afrl-mitll-backtrans.4907.en-ru.sgm"]=0.655
                 ["newstest2017.jhu-pbmt.4986.en-ru.sgm"]=0.658
                 ["newstest2017.online-A.0.en-ru.sgm"]=0.661
                 ["newstest2017.online-B.0.en-ru.sgm"]=0.563
                 ["newstest2017.online-F.0.en-ru.sgm"]=0.809
                 ["newstest2017.online-G.0.en-ru.sgm"]=0.618
                 ["newstest2017.PROMT-Rule-based.4736.en-ru.sgm"]=0.667
                 ["newstest2017.uedin-nmt.4756.en-ru.sgm"]=0.614
                 ["newstest2017.JAIST.4858.en-tr.sgm"]=0.875
                 ["newstest2017.jhu-nmt-lattice-rescore.4904.en-tr.sgm"]=0.864
                 ["newstest2017.jhu-pbmt.4970.en-tr.sgm"]=0.874
                 ["newstest2017.LIUM-NMT.4953.en-tr.sgm"]=0.752
                 ["newstest2017.online-A.0.en-tr.sgm"]=0.787
                 ["newstest2017.online-B.0.en-tr.sgm"]=0.677
                 ["newstest2017.uedin-nmt.4932.en-tr.sgm"]=0.747
                 ["newstest2017.CASICT-DCU-NMT.5157.en-zh.sgm"]=0.999
                 ["newstest2017.online-A.0.en-zh.sgm"]=2.936
                 ["newstest2017.online-B.0.en-zh.sgm"]=1.092
                 ["newstest2017.online-F.0.en-zh.sgm"]=1.114
                 ["newstest2017.online-G.0.en-zh.sgm"]=1.158
                 ["newstest2017.Oregon-State-University-S.5174.en-zh.sgm"]=5.622
                 ["newstest2017.SogouKnowing-nmt.5131.en-zh.sgm"]=1.402
                 ["newstest2017.uedin-nmt.5111.en-zh.sgm"]=1.334
                 ["newstest2017.UU-HNMT.5134.en-zh.sgm"]=1.000
                 ["newstest2017.xmunmt.5165.en-zh.sgm"]=1.366
                  )

if [ -z $SKIP_TER ]; then
  echo "-------------------"
  echo "Starting TER tests"
  echo "-------------------"
  for pair in cs-en en-cs en-de en-fi en-lv en-ru en-tr en-zh; do
      source=$(echo $pair | cut -d- -f1)
      target=$(echo $pair | cut -d- -f2)
      for sgm in wmt17-submitted-data/sgm/system-outputs/newstest2017/$pair/*.sgm; do
          name=$(basename $sgm)

          if [[ ! -z $limit_test && $limit_test != $name ]]; then continue; fi

          if [[ ! -v "TER[$name]" ]]; then continue ; fi  # official Tercom fails for some outputs...

          sys=$(basename $sgm .sgm | perl -pe 's/newstest2017\.//')
          txt=$(dirname $sgm | perl -pe 's/sgm/txt/')/$(basename $sgm .sgm)
          src=wmt17-submitted-data/sgm/sources/newstest2017-$source$target-src.$source.sgm
          ref=wmt17-submitted-data/sgm/references/newstest2017-$source$target-ref.$target.sgm

          score=$(cat $txt | ${CMD} -w 3 -t wmt17 -l $source-$target -b --metrics ter)

          expected_score="${TER[$name]}"

          echo "import sys; sys.exit(1 if abs(${score}-${expected_score}) > 0.01 else 0)" | python

          if [[ $? -eq 1 ]]; then
              echo "FAILED test $pair/$sys (wanted $expected_score got $score)"
              exit 1
          fi
          echo "Passed $source-$target $sys tercom: $expected_score sacreTER: $score"

          let i++
      done
  done
fi

#############
# Mecab tests
#############
if [[ -z $SKIP_MECAB ]]; then
  echo "-----------------------"
  echo "Testing Mecab tokenizer"
  echo "-----------------------"
    pair="en-ja"
    source=$(echo $pair | cut -d- -f1)
    target=$(echo $pair | cut -d- -f2)
    for txt in en-ja-translation-example-master/*.hyp.$target; do
        name=$(basename $txt .hyp.$target)

        if [[ ! -z $limit_test && $limit_test != $name ]]; then continue; fi

        sys=$(basename $txt .hyp.$target)
        ref=$(dirname $txt)/$(basename $txt .hyp.$target).ref.$target
        score=$(cat $txt | ${CMD} -w 2 -l $source-$target -b $ref)

        echo "import sys; sys.exit(1 if abs($score-${MTEVAL[$name]}) > 0.01 else 0)" | python

        if [[ $? -eq 1 ]]; then
            echo "FAILED test $pair/$sys (wanted ${MTEVAL[$name]} got $score)"
            exit 1
        fi
        echo "Passed $source-$target $sys mteval-v13a.pl: ${MTEVAL[$name]} sacreBLEU: $score"

        let i++
    done
fi

echo "Passed $i tests."
exit 0
