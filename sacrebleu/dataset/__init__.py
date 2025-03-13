#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017--2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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


# This defines data locations.
# Right below are test sets.
# Beneath each test set, we define the location to download the test data.
# The other keys are each language pair contained in the tarball, and the respective locations of the source and reference data within each.
# Many of these are *.sgm files, which are processed to produced plain text that can be used by this script.
# The canonical location of unpacked, processed data is $SACREBLEU_DIR/$TEST/$SOURCE-$TARGET.{$SOURCE,$TARGET}
from .fake_sgml import FakeSGMLDataset, WMTAdditionDataset
from .iwslt_xml import IWSLTXMLDataset
from .plain_text import PlainTextDataset
from .tsv import TSVDataset
from .wmt_xml import WMTXMLDataset

# Detailed document metadata annotation in form DocumentID -> CountryCode - Domain - OptionalFinegrainedCountryCode
# While the annotation is subjective with many unclear cases, it may provide useful insights
# when applied on large data (TODO: annotate all documents from recent WMT years, at least for origlang=en, consider renaming "world" to "other").
_SUBSETS = {
    "wmt18": "rt.com.68098=US-crime guardian.181611=US-politics bbc.310963=GB-sport washpost.116881=US-politics scotsman.104228=GB-sport timemagazine.75207=OTHER-world-ID "
    "euronews-en.117981=OTHER-crime-AE smh.com.au.242810=US-crime msnbc.53726=US-politics euronews-en.117983=US-politics msnbc.53894=US-crime theglobeandmail.com.62700=US-business "
    "bbc.310870=OTHER-world-AF reuters.196698=US-politics latimes.231739=US-sport thelocal.51929=OTHER-world-SE cbsnews.198694=US-politics reuters.196718=OTHER-sport-RU "
    "abcnews.255599=EU-sport nytimes.127256=US-entertainment scotsman.104225=GB-politics dailymail.co.uk.233026=GB-scitech independent.181088=GB-entertainment "
    "brisbanetimes.com.au.181614=OTHER-business-AU washpost.116837=US-politics dailymail.co.uk.232928=GB-world thelocal.51916=OTHER-politics-IT bbc.310871=US-crime "
    "nytimes.127392=EU-business-DE euronews-en.118001=EU-scitech-FR washpost.116866=OTHER-crime-MX dailymail.co.uk.233025=OTHER-scitech-CA latimes.231829=US-crime "
    "guardian.181662=US-entertainment msnbc.53731=US-crime rt.com.68127=OTHER-sport-RU latimes.231782=US-business latimes.231840=US-sport reuters.196711=OTHER-scitech "
    "guardian.181666=GB-entertainment novinite.com.24019=US-politics smh.com.au.242750=OTHER-scitech guardian.181610=US-politics telegraph.364393=OTHER-crime-ZA "
    "novinite.com.23995=EU-world dailymail.co.uk.233028=GB-scitech independent.181071=GB-sport telegraph.364538=GB-scitech timemagazine.75193=US-politics "
    "independent.181096=US-entertainment upi.140602=OTHER-world-AF bbc.310946=GB-business independent.181052=EU-sport ",
    "wmt19": "bbc.381790=GB-politics rt.com.91337=OTHER-politics-MK nytimes.184853=US-world upi.176266=US-crime guardian.221754=GB-business dailymail.co.uk.298595=GB-business "
    "cnbc.com.6790=US-politics nytimes.184837=OTHER-world-ID upi.176249=GB-sport euronews-en.153835=OTHER-world-ID dailymail.co.uk.298732=GB-crime telegraph.405401=GB-politics "
    "newsweek.51331=OTHER-crime-CN abcnews.306815=US-world cbsnews.248384=US-politics reuters.218882=GB-politics cbsnews.248387=US-crime abcnews.306764=OTHER-world-MX "
    "reuters.218888=EU-politics bbc.381780=GB-crime bbc.381746=GB-sport euronews-en.153800=EU-politics bbc.381679=GB-crime bbc.381735=GB-crime newsweek.51338=US-world "
    "bbc.381765=GB-crime cnn.304489=US-politics reuters.218863=OTHER-world-ID nytimes.184860=OTHER-world-ID cnn.304404=US-crime bbc.381647=US-entertainment "
    "abcnews.306758=OTHER-politics-MX cnbc.com.6772=US-business reuters.218932=OTHER-politics-MK upi.176251=GB-sport reuters.218921=US-sport cnn.304447=US-politics "
    "guardian.221679=GB-politics scotsman.133765=GB-sport scotsman.133804=GB-entertainment guardian.221762=OTHER-politics-BO cnbc.com.6769=US-politics "
    "dailymail.co.uk.298692=EU-entertainment scotsman.133744=GB-world reuters.218911=US-sport newsweek.51310=US-politics independent.226301=US-sport reuters.218923=EU-sport "
    "reuters.218861=US-politics dailymail.co.uk.298759=US-world scotsman.133791=GB-sport cbsnews.248484=EU-scitech dailymail.co.uk.298630=US-scitech "
    "newsweek.51329=US-entertainment bbc.381701=GB-crime dailymail.co.uk.298738=GB-entertainment bbc.381669=OTHER-world-CN foxnews.94512=US-politics "
    "guardian.221718=GB-entertainment dailymail.co.uk.298686=GB-politics cbsnews.248471=US-politics newsweek.51318=US-entertainment rt.com.91335=US-politics "
    "newsweek.51300=US-politics cnn.304478=US-politics upi.176275=US-politics telegraph.405422=OTHER-world-ID reuters.218933=US-politics newsweek.51328=US-politics "
    "newsweek.51307=US-business bbc.381692=GB-world independent.226346=GB-entertainment bbc.381646=GB-sport reuters.218914=US-sport scotsman.133758=EU-sport "
    "rt.com.91350=EU-world scotsman.133773=GB-scitech rt.com.91334=EU-crime bbc.381680=GB-politics guardian.221756=US-politics scotsman.133783=GB-politics cnn.304521=US-sport "
    "dailymail.co.uk.298622=GB-politics bbc.381789=GB-sport dailymail.co.uk.298644=GB-business dailymail.co.uk.298602=GB-world scotsman.133753=GB-sport "
    "independent.226317=GB-entertainment nytimes.184862=US-politics thelocal.65969=OTHER-world-SY nytimes.184825=US-politics cnbc.com.6784=US-politics nytimes.184804=US-politics "
    "nytimes.184830=US-politics scotsman.133801=GB-sport cnbc.com.6770=US-business bbc.381760=GB-crime reuters.218865=OTHER-world-ID newsweek.51339=US-crime "
    "euronews-en.153797=OTHER-world-ID abcnews.306774=US-crime dailymail.co.uk.298696=GB-politics abcnews.306755=US-politics reuters.218909=US-crime "
    "independent.226349=OTHER-sport-RU newsweek.51330=US-politics bbc.381705=GB-sport newsweek.51340=OTHER-world-ID cbsnews.248411=OTHER-world-FM abcnews.306776=US-crime "
    "bbc.381694=GB-entertainment rt.com.91356=US-world telegraph.405430=GB-entertainment telegraph.405404=EU-world bbc.381749=GB-world telegraph.405413=US-politics "
    "bbc.381736=OTHER-politics-KP cbsnews.248394=US-politics nytimes.184822=US-world telegraph.405408=US-politics euronews-en.153799=OTHER-politics-SY "
    "euronews-en.153826=EU-sport cnn.304400=US-world",
}

SUBSETS = {
    k: {d.split("=")[0]: d.split("=")[1] for d in v.split()}
    for (k, v) in _SUBSETS.items()
}
COUNTRIES = sorted(list({v.split("-")[0] for v in SUBSETS["wmt19"].values()}))
DOMAINS = sorted(list({v.split("-")[1] for v in SUBSETS["wmt19"].values()}))

DATASETS = {
    # wmt
    "wmt24": WMTXMLDataset(
        "wmt24",
        data=["https://github.com/wmt-conference/wmt24-news-systems/releases/download/v1.1/data_onlyxml.tar.gz"],
        description="WMT24 official test set release, v1.1 (excluding TS)",
        md5=["299963fcb7b4e86d6d212bf69beb9580"],
        langpairs={
            "cs-uk": ["xml/wmttest2024.cs-uk.all.xml"],
            "en-cs": ["xml/wmttest2024.en-cs.all.xml"],
            "en-de": ["xml/wmttest2024.en-de.all.xml"],
            "en-es": ["xml/wmttest2024.en-es.all.xml"],
            "en-hi": ["xml/wmttest2024.en-hi.all.xml"],
            "en-is": ["xml/wmttest2024.en-is.all.xml"],
            "en-ja": ["xml/wmttest2024.en-ja.all.xml"],
            "en-ru": ["xml/wmttest2024.en-ru.all.xml"],
            "en-uk": ["xml/wmttest2024.en-uk.all.xml"],
            "en-zh": ["xml/wmttest2024.en-zh.all.xml"],
            "ja-zh": ["xml/wmttest2024.ja-zh.all.xml"],
        },
        refs=["refA"],
    ),
    "wmt23": WMTXMLDataset(
        "wmt23",
        data=["https://github.com/wmt-conference/wmt23-news-systems/archive/refs/tags/v.0.1.tar.gz"],
        description="Official evaluation and system data for WMT23.",
        md5=["63576405e4ce07130a19ad76ba7eb75b"],
        langpairs={
            "cs-uk": ["wmt23-news-systems-v.0.1/xml/wmttest2023.cs-uk.all.xml"],
            "de-en": ["wmt23-news-systems-v.0.1/xml/wmttest2023.de-en.all.xml"],
            "en-cs": ["wmt23-news-systems-v.0.1/xml/wmttest2023.en-cs.all.xml"],
            "en-de": ["wmt23-news-systems-v.0.1/xml/wmttest2023.en-de.all.xml"],
            "en-he": {
                "path": "wmt23-news-systems-v.0.1/xml/wmttest2023.en-he.all.xml",
                "refs": ["refB"],
            },
            "en-ja": ["wmt23-news-systems-v.0.1/xml/wmttest2023.en-ja.all.xml"],
            "en-ru": ["wmt23-news-systems-v.0.1/xml/wmttest2023.en-ru.all.xml"],
            "en-uk": ["wmt23-news-systems-v.0.1/xml/wmttest2023.en-uk.all.xml"],
            "en-zh": ["wmt23-news-systems-v.0.1/xml/wmttest2023.en-zh.all.xml"],
            "he-en": {
                "path": "wmt23-news-systems-v.0.1/xml/wmttest2023.he-en.all.xml",
                "refs": ["refB"],
            },
            "ja-en": ["wmt23-news-systems-v.0.1/xml/wmttest2023.ja-en.all.xml"],
            "ru-en": ["wmt23-news-systems-v.0.1/xml/wmttest2023.ru-en.all.xml"],
            "uk-en": ["wmt23-news-systems-v.0.1/xml/wmttest2023.uk-en.all.xml"],
            "zh-en": ["wmt23-news-systems-v.0.1/xml/wmttest2023.zh-en.all.xml"],
        },
        refs=["refA"],
    ),
    "wmt22": WMTXMLDataset(
        "wmt22",
        data=["https://github.com/wmt-conference/wmt22-news-systems/archive/refs/tags/v1.1.tar.gz"],
        description="Official evaluation and system data for WMT22.",
        md5=["0840978b9b50b9ac3b2b081e37d620b9"],
        langpairs={
            "cs-en": {
                "path": "wmt22-news-systems-1.1/xml/wmttest2022.cs-en.all.xml",
                "refs": ["B"],
            },
            "cs-uk": ["wmt22-news-systems-1.1/xml/wmttest2022.cs-uk.all.xml"],
            "de-en": ["wmt22-news-systems-1.1/xml/wmttest2022.de-en.all.xml"],
            "de-fr": ["wmt22-news-systems-1.1/xml/wmttest2022.de-fr.all.xml"],
            "en-cs": {
                "path": "wmt22-news-systems-1.1/xml/wmttest2022.en-cs.all.xml",
                "refs": ["B"],
            },
            "en-de": ["wmt22-news-systems-1.1/xml/wmttest2022.en-de.all.xml"],
            "en-hr": ["wmt22-news-systems-1.1/xml/wmttest2022.en-hr.all.xml"],
            "en-ja": ["wmt22-news-systems-1.1/xml/wmttest2022.en-ja.all.xml"],
            "en-liv": ["wmt22-news-systems-1.1/xml/wmttest2022.en-liv.all.xml"],
            "en-ru": ["wmt22-news-systems-1.1/xml/wmttest2022.en-ru.all.xml"],
            "en-uk": ["wmt22-news-systems-1.1/xml/wmttest2022.en-uk.all.xml"],
            "en-zh": ["wmt22-news-systems-1.1/xml/wmttest2022.en-zh.all.xml"],
            "fr-de": ["wmt22-news-systems-1.1/xml/wmttest2022.fr-de.all.xml"],
            "ja-en": ["wmt22-news-systems-1.1/xml/wmttest2022.ja-en.all.xml"],
            "liv-en": ["wmt22-news-systems-1.1/xml/wmttest2022.liv-en.all.xml"],
            "ru-en": ["wmt22-news-systems-1.1/xml/wmttest2022.ru-en.all.xml"],
            "ru-sah": ["wmt22-news-systems-1.1/xml/wmttest2022.ru-sah.all.xml"],
            "sah-ru": ["wmt22-news-systems-1.1/xml/wmttest2022.sah-ru.all.xml"],
            "uk-cs": ["wmt22-news-systems-1.1/xml/wmttest2022.uk-cs.all.xml"],
            "uk-en": ["wmt22-news-systems-1.1/xml/wmttest2022.uk-en.all.xml"],
            "zh-en": ["wmt22-news-systems-1.1/xml/wmttest2022.zh-en.all.xml"],
        },
        # the default reference to use with this dataset
        refs=["A"],
    ),
    "wmt21/systems": WMTXMLDataset(
        "wmt21/systems",
        data=["https://github.com/wmt-conference/wmt21-news-systems/archive/refs/tags/v1.3.tar.gz"],
        description="WMT21 system output.",
        md5=["a6aee4099da58f98f71eb3fac1694237"],
        langpairs={
            "de-fr": ["wmt21-news-systems-1.3/xml/newstest2021.de-fr.all.xml"],
            "en-de": ["wmt21-news-systems-1.3/xml/newstest2021.en-de.all.xml"],
            "en-ha": ["wmt21-news-systems-1.3/xml/newstest2021.en-ha.all.xml"],
            "en-is": ["wmt21-news-systems-1.3/xml/newstest2021.en-is.all.xml"],
            "en-ja": ["wmt21-news-systems-1.3/xml/newstest2021.en-ja.all.xml"],
            "fr-de": ["wmt21-news-systems-1.3/xml/newstest2021.fr-de.all.xml"],
            "ha-en": ["wmt21-news-systems-1.3/xml/newstest2021.ha-en.all.xml"],
            "is-en": ["wmt21-news-systems-1.3/xml/newstest2021.is-en.all.xml"],
            "ja-en": ["wmt21-news-systems-1.3/xml/newstest2021.ja-en.all.xml"],
            "zh-en": ["wmt21-news-systems-1.3/xml/newstest2021.zh-en.all.xml"],
            "en-zh": ["wmt21-news-systems-1.3/xml/newstest2021.en-zh.all.xml"],
            "cs-en": ["wmt21-news-systems-1.3/xml/newstest2021.cs-en.all.xml"],
            "de-en": ["wmt21-news-systems-1.3/xml/newstest2021.de-en.all.xml"],
            "en-cs": ["wmt21-news-systems-1.3/xml/newstest2021.en-cs.all.xml"],
            "en-ru": ["wmt21-news-systems-1.3/xml/newstest2021.en-ru.all.xml"],
            "ru-en": ["wmt21-news-systems-1.3/xml/newstest2021.ru-en.all.xml"],
            "bn-hi": ["wmt21-news-systems-1.3/xml/florestest2021.bn-hi.all.xml"],
            "hi-bn": ["wmt21-news-systems-1.3/xml/florestest2021.hi-bn.all.xml"],
            "xh-zu": ["wmt21-news-systems-1.3/xml/florestest2021.xh-zu.all.xml"],
            "zu-xh": ["wmt21-news-systems-1.3/xml/florestest2021.zu-xh.all.xml"],
        },
        # the reference to use with this dataset
        refs=["A"],
    ),
    "wmt21": WMTXMLDataset(
        "wmt21",
        data=["https://data.statmt.org/wmt21/translation-task/test.tgz"],
        description="Official evaluation data for WMT21.",
        md5=["32e7ab995bc318414375d60f0269af92"],
        langpairs={
            "de-fr": ["test/newstest2021.de-fr.xml"],
            "en-de": ["test/newstest2021.en-de.xml"],
            "en-ha": ["test/newstest2021.en-ha.xml"],
            "en-is": ["test/newstest2021.en-is.xml"],
            "en-ja": ["test/newstest2021.en-ja.xml"],
            "fr-de": ["test/newstest2021.fr-de.xml"],
            "ha-en": ["test/newstest2021.ha-en.xml"],
            "is-en": ["test/newstest2021.is-en.xml"],
            "ja-en": ["test/newstest2021.ja-en.xml"],
            "zh-en": ["test/newstest2021.zh-en.xml"],
            "en-zh": ["test/newstest2021.en-zh.xml"],
            "cs-en": ["test/newstest2021.cs-en.xml"],
            "de-en": ["test/newstest2021.de-en.xml"],
            "en-cs": ["test/newstest2021.en-cs.xml"],
            "en-ru": ["test/newstest2021.en-ru.xml"],
            "ru-en": ["test/newstest2021.ru-en.xml"],
            "bn-hi": ["test/florestest2021.bn-hi.xml"],
            "hi-bn": ["test/florestest2021.hi-bn.xml"],
            "xh-zu": ["test/florestest2021.xh-zu.xml"],
            "zu-xh": ["test/florestest2021.zu-xh.xml"],
        },
        # the reference to use with this dataset
        refs=["A"],
    ),
    "wmt21/B": WMTXMLDataset(
        "wmt21/B",
        data=["https://data.statmt.org/wmt21/translation-task/test.tgz"],
        description="Official evaluation data for WMT21 with reference B.",
        md5=["32e7ab995bc318414375d60f0269af92"],
        langpairs={
            "cs-en": ["test/newstest2021.cs-en.xml"],
            "de-en": ["test/newstest2021.de-en.xml"],
            "en-cs": ["test/newstest2021.en-cs.xml"],
            "en-ru": ["test/newstest2021.en-ru.xml"],
            "en-zh": ["test/newstest2021.en-zh.xml"],
            "ru-en": ["test/newstest2021.ru-en.xml"],
        },
        # the reference to use with this dataset
        refs=["B"],
    ),
    "wmt21/AB": WMTXMLDataset(
        "wmt21/AB",
        data=["https://data.statmt.org/wmt21/translation-task/test.tgz"],
        description="Official evaluation data for WMT21 with references A and B.",
        md5=["32e7ab995bc318414375d60f0269af92"],
        langpairs={
            "cs-en": ["test/newstest2021.cs-en.xml"],
            "de-en": ["test/newstest2021.de-en.xml"],
            "en-de": ["test/newstest2021.en-de.xml"],
            "en-cs": ["test/newstest2021.en-cs.xml"],
            "en-ru": ["test/newstest2021.en-ru.xml"],
            "en-zh": ["test/newstest2021.en-zh.xml"],
            "ru-en": ["test/newstest2021.ru-en.xml"],
        },
        # the reference to use with this dataset
        refs=["A", "B"],
    ),
    "wmt21/C": WMTXMLDataset(
        "wmt21/C",
        data=["https://data.statmt.org/wmt21/translation-task/test.tgz"],
        description="Official evaluation data for WMT21 with reference C",
        md5=["32e7ab995bc318414375d60f0269af92"],
        langpairs={
            "en-de": ["test/newstest2021.en-de.xml"],
        },
        # the reference to use with this dataset
        refs=["C"],
    ),
    "wmt21/AC": WMTXMLDataset(
        "wmt21/AC",
        data=["https://data.statmt.org/wmt21/translation-task/test.tgz"],
        description="Official evaluation data for WMT21 with references A and C",
        md5=["32e7ab995bc318414375d60f0269af92"],
        langpairs={
            "en-de": ["test/newstest2021.en-de.xml"],
        },
        # the reference to use with this dataset
        refs=["A", "C"],
    ),
    "wmt21/D": WMTXMLDataset(
        "wmt21/D",
        data=["https://data.statmt.org/wmt21/translation-task/test.tgz"],
        description="Official evaluation data for WMT21 with reference D",
        md5=["32e7ab995bc318414375d60f0269af92"],
        langpairs={
            "en-de": ["test/newstest2021.en-de.xml"],
        },
        # the reference to use with this dataset
        refs=["D"],
    ),
    "wmt21/dev": WMTXMLDataset(
        "wmt21/dev",
        data=["https://data.statmt.org/wmt21/translation-task/dev.tgz"],
        description="Development data for WMT21，if multiple references are available, the first one is used.",
        md5=["165da59ac8dfb5b7cafd7e90b1cac672"],
        langpairs={
            "en-ha": ["dev/xml/newsdev2021.en-ha.xml"],
            "ha-en": ["dev/xml/newsdev2021.ha-en.xml"],
            "en-is": ["dev/xml/newsdev2021.en-is.xml"],
            "is-en": ["dev/xml/newsdev2021.is-en.xml"],
        },
        # datasets are bidirectional in origin, so use both refs
        refs=["A", ""],
    ),
    "wmt20/tworefs": FakeSGMLDataset(
        "wmt20/tworefs",
        data=["https://data.statmt.org/wmt20/translation-task/test.tgz"],
        description="WMT20 news test sets with two references",
        md5=["3b1f777cfd2fb15ccf66e9bfdb2b1699"],
        langpairs={
            "de-en": [
                "sgm/newstest2020-deen-src.de.sgm",
                "sgm/newstest2020-deen-ref.en.sgm",
                "sgm/newstestB2020-deen-ref.en.sgm",
            ],
            "en-de": [
                "sgm/newstest2020-ende-src.en.sgm",
                "sgm/newstest2020-ende-ref.de.sgm",
                "sgm/newstestB2020-ende-ref.de.sgm",
            ],
            "en-zh": [
                "sgm/newstest2020-enzh-src.en.sgm",
                "sgm/newstest2020-enzh-ref.zh.sgm",
                "sgm/newstestB2020-enzh-ref.zh.sgm",
            ],
            "ru-en": [
                "sgm/newstest2020-ruen-src.ru.sgm",
                "sgm/newstest2020-ruen-ref.en.sgm",
                "sgm/newstestB2020-ruen-ref.en.sgm",
            ],
            "zh-en": [
                "sgm/newstest2020-zhen-src.zh.sgm",
                "sgm/newstest2020-zhen-ref.en.sgm",
                "sgm/newstestB2020-zhen-ref.en.sgm",
            ],
        },
    ),
    "wmt20": FakeSGMLDataset(
        "wmt20",
        data=["https://data.statmt.org/wmt20/translation-task/test.tgz"],
        description="Official evaluation data for WMT20",
        md5=["3b1f777cfd2fb15ccf66e9bfdb2b1699"],
        langpairs={
            "cs-en": [
                "sgm/newstest2020-csen-src.cs.sgm",
                "sgm/newstest2020-csen-ref.en.sgm",
            ],
            "de-en": [
                "sgm/newstest2020-deen-src.de.sgm",
                "sgm/newstest2020-deen-ref.en.sgm",
            ],
            "de-fr": [
                "sgm/newstest2020-defr-src.de.sgm",
                "sgm/newstest2020-defr-ref.fr.sgm",
            ],
            "en-cs": [
                "sgm/newstest2020-encs-src.en.sgm",
                "sgm/newstest2020-encs-ref.cs.sgm",
            ],
            "en-de": [
                "sgm/newstest2020-ende-src.en.sgm",
                "sgm/newstest2020-ende-ref.de.sgm",
            ],
            "en-iu": [
                "sgm/newstest2020-eniu-src.en.sgm",
                "sgm/newstest2020-eniu-ref.iu.sgm",
            ],
            "en-ja": [
                "sgm/newstest2020-enja-src.en.sgm",
                "sgm/newstest2020-enja-ref.ja.sgm",
            ],
            "en-km": [
                "sgm/newstest2020-enkm-src.en.sgm",
                "sgm/newstest2020-enkm-ref.km.sgm",
            ],
            "en-pl": [
                "sgm/newstest2020-enpl-src.en.sgm",
                "sgm/newstest2020-enpl-ref.pl.sgm",
            ],
            "en-ps": [
                "sgm/newstest2020-enps-src.en.sgm",
                "sgm/newstest2020-enps-ref.ps.sgm",
            ],
            "en-ru": [
                "sgm/newstest2020-enru-src.en.sgm",
                "sgm/newstest2020-enru-ref.ru.sgm",
            ],
            "en-ta": [
                "sgm/newstest2020-enta-src.en.sgm",
                "sgm/newstest2020-enta-ref.ta.sgm",
            ],
            "en-zh": [
                "sgm/newstest2020-enzh-src.en.sgm",
                "sgm/newstest2020-enzh-ref.zh.sgm",
            ],
            "fr-de": [
                "sgm/newstest2020-frde-src.fr.sgm",
                "sgm/newstest2020-frde-ref.de.sgm",
            ],
            "iu-en": [
                "sgm/newstest2020-iuen-src.iu.sgm",
                "sgm/newstest2020-iuen-ref.en.sgm",
            ],
            "ja-en": [
                "sgm/newstest2020-jaen-src.ja.sgm",
                "sgm/newstest2020-jaen-ref.en.sgm",
            ],
            "km-en": [
                "sgm/newstest2020-kmen-src.km.sgm",
                "sgm/newstest2020-kmen-ref.en.sgm",
            ],
            "pl-en": [
                "sgm/newstest2020-plen-src.pl.sgm",
                "sgm/newstest2020-plen-ref.en.sgm",
            ],
            "ps-en": [
                "sgm/newstest2020-psen-src.ps.sgm",
                "sgm/newstest2020-psen-ref.en.sgm",
            ],
            "ru-en": [
                "sgm/newstest2020-ruen-src.ru.sgm",
                "sgm/newstest2020-ruen-ref.en.sgm",
            ],
            "ta-en": [
                "sgm/newstest2020-taen-src.ta.sgm",
                "sgm/newstest2020-taen-ref.en.sgm",
            ],
            "zh-en": [
                "sgm/newstest2020-zhen-src.zh.sgm",
                "sgm/newstest2020-zhen-ref.en.sgm",
            ],
        },
    ),
    "wmt20/dev": FakeSGMLDataset(
        "wmt20/dev",
        data=["https://data.statmt.org/wmt20/translation-task/dev.tgz"],
        description="Development data for tasks new to 2020.",
        md5=["037f2b37aab74febbb1b2307dc2afb54"],
        langpairs={
            "iu-en": [
                "dev/newsdev2020-iuen-src.iu.sgm",
                "dev/newsdev2020-iuen-ref.en.sgm",
            ],
            "en-iu": [
                "dev/newsdev2020-eniu-src.en.sgm",
                "dev/newsdev2020-eniu-ref.iu.sgm",
            ],
            "ja-en": [
                "dev/newsdev2020-jaen-src.ja.sgm",
                "dev/newsdev2020-jaen-ref.en.sgm",
            ],
            "en-ja": [
                "dev/newsdev2020-enja-src.en.sgm",
                "dev/newsdev2020-enja-ref.ja.sgm",
            ],
            "pl-en": [
                "dev/newsdev2020-plen-src.pl.sgm",
                "dev/newsdev2020-plen-ref.en.sgm",
            ],
            "en-pl": [
                "dev/newsdev2020-enpl-src.en.sgm",
                "dev/newsdev2020-enpl-ref.pl.sgm",
            ],
            "ta-en": [
                "dev/newsdev2020-taen-src.ta.sgm",
                "dev/newsdev2020-taen-ref.en.sgm",
            ],
            "en-ta": [
                "dev/newsdev2020-enta-src.en.sgm",
                "dev/newsdev2020-enta-ref.ta.sgm",
            ],
        },
    ),
    "wmt20/robust/set1": PlainTextDataset(
        "wmt20/robust/set1",
        data=["https://data.statmt.org/wmt20/robustness-task/robustness20-3-sets.zip"],
        md5=["a12ac9ebe89b72195041518dffc4a9d5"],
        description="WMT20 robustness task, set 1",
        langpairs={
            "en-ja": [
                "robustness20-3-sets/robustness20-set1-enja.en",
                "robustness20-3-sets/robustness20-set1-enja.ja",
            ],
            "en-de": [
                "robustness20-3-sets/robustness20-set1-ende.en",
                "robustness20-3-sets/robustness20-set1-ende.de",
            ],
        },
    ),
    "wmt20/robust/set2": PlainTextDataset(
        "wmt20/robust/set2",
        data=["https://data.statmt.org/wmt20/robustness-task/robustness20-3-sets.zip"],
        md5=["a12ac9ebe89b72195041518dffc4a9d5"],
        description="WMT20 robustness task, set 2",
        langpairs={
            "en-ja": [
                "robustness20-3-sets/robustness20-set2-enja.en",
                "robustness20-3-sets/robustness20-set2-enja.ja",
            ],
            "ja-en": [
                "robustness20-3-sets/robustness20-set2-jaen.ja",
                "robustness20-3-sets/robustness20-set2-jaen.en",
            ],
        },
    ),
    "wmt20/robust/set3": PlainTextDataset(
        "wmt20/robust/set3",
        data=["https://data.statmt.org/wmt20/robustness-task/robustness20-3-sets.zip"],
        md5=["a12ac9ebe89b72195041518dffc4a9d5"],
        description="WMT20 robustness task, set 3",
        langpairs={
            "de-en": [
                "robustness20-3-sets/robustness20-set3-deen.de",
                "robustness20-3-sets/robustness20-set3-deen.en",
            ],
        },
    ),
    "wmt19": FakeSGMLDataset(
        "wmt19",
        data=["https://data.statmt.org/wmt19/translation-task/test.tgz"],
        description="Official evaluation data.",
        md5=["84de7162d158e28403103b01aeefc39a"],
        citation=r"""@proceedings{ws-2019-machine,
    title = "Proceedings of the Fourth Conference on Machine Translation (Volume 1: Research Papers)",
    editor = "Bojar, Ond{\v{r}}ej  and
      Chatterjee, Rajen  and
      Federmann, Christian  and
      Fishel, Mark  and
      Graham, Yvette  and
      Haddow, Barry  and
      Huck, Matthias  and
      Yepes, Antonio Jimeno  and
      Koehn, Philipp  and
      Martins, Andr{\'e}  and
      Monz, Christof  and
      Negri, Matteo  and
      N{\'e}v{\'e}ol, Aur{\'e}lie  and
      Neves, Mariana  and
      Post, Matt  and
      Turchi, Marco  and
      Verspoor, Karin",
    month = aug,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W19-5200",
}""",
        langpairs={
            "cs-de": [
                "sgm/newstest2019-csde-src.cs.sgm",
                "sgm/newstest2019-csde-ref.de.sgm",
            ],
            "de-cs": [
                "sgm/newstest2019-decs-src.de.sgm",
                "sgm/newstest2019-decs-ref.cs.sgm",
            ],
            "de-en": [
                "sgm/newstest2019-deen-src.de.sgm",
                "sgm/newstest2019-deen-ref.en.sgm",
            ],
            "de-fr": [
                "sgm/newstest2019-defr-src.de.sgm",
                "sgm/newstest2019-defr-ref.fr.sgm",
            ],
            "en-cs": [
                "sgm/newstest2019-encs-src.en.sgm",
                "sgm/newstest2019-encs-ref.cs.sgm",
            ],
            "en-de": [
                "sgm/newstest2019-ende-src.en.sgm",
                "sgm/newstest2019-ende-ref.de.sgm",
            ],
            "en-fi": [
                "sgm/newstest2019-enfi-src.en.sgm",
                "sgm/newstest2019-enfi-ref.fi.sgm",
            ],
            "en-gu": [
                "sgm/newstest2019-engu-src.en.sgm",
                "sgm/newstest2019-engu-ref.gu.sgm",
            ],
            "en-kk": [
                "sgm/newstest2019-enkk-src.en.sgm",
                "sgm/newstest2019-enkk-ref.kk.sgm",
            ],
            "en-lt": [
                "sgm/newstest2019-enlt-src.en.sgm",
                "sgm/newstest2019-enlt-ref.lt.sgm",
            ],
            "en-ru": [
                "sgm/newstest2019-enru-src.en.sgm",
                "sgm/newstest2019-enru-ref.ru.sgm",
            ],
            "en-zh": [
                "sgm/newstest2019-enzh-src.en.sgm",
                "sgm/newstest2019-enzh-ref.zh.sgm",
            ],
            "fi-en": [
                "sgm/newstest2019-fien-src.fi.sgm",
                "sgm/newstest2019-fien-ref.en.sgm",
            ],
            "fr-de": [
                "sgm/newstest2019-frde-src.fr.sgm",
                "sgm/newstest2019-frde-ref.de.sgm",
            ],
            "gu-en": [
                "sgm/newstest2019-guen-src.gu.sgm",
                "sgm/newstest2019-guen-ref.en.sgm",
            ],
            "kk-en": [
                "sgm/newstest2019-kken-src.kk.sgm",
                "sgm/newstest2019-kken-ref.en.sgm",
            ],
            "lt-en": [
                "sgm/newstest2019-lten-src.lt.sgm",
                "sgm/newstest2019-lten-ref.en.sgm",
            ],
            "ru-en": [
                "sgm/newstest2019-ruen-src.ru.sgm",
                "sgm/newstest2019-ruen-ref.en.sgm",
            ],
            "zh-en": [
                "sgm/newstest2019-zhen-src.zh.sgm",
                "sgm/newstest2019-zhen-ref.en.sgm",
            ],
        },
    ),
    "wmt19/dev": FakeSGMLDataset(
        "wmt19/dev",
        data=["https://data.statmt.org/wmt19/translation-task/dev.tgz"],
        description="Development data for tasks new to 2019.",
        md5=["f2ec7af5947c19e0cacb3882eb208002"],
        langpairs={
            "lt-en": [
                "dev/newsdev2019-lten-src.lt.sgm",
                "dev/newsdev2019-lten-ref.en.sgm",
            ],
            "en-lt": [
                "dev/newsdev2019-enlt-src.en.sgm",
                "dev/newsdev2019-enlt-ref.lt.sgm",
            ],
            "gu-en": [
                "dev/newsdev2019-guen-src.gu.sgm",
                "dev/newsdev2019-guen-ref.en.sgm",
            ],
            "en-gu": [
                "dev/newsdev2019-engu-src.en.sgm",
                "dev/newsdev2019-engu-ref.gu.sgm",
            ],
            "kk-en": [
                "dev/newsdev2019-kken-src.kk.sgm",
                "dev/newsdev2019-kken-ref.en.sgm",
            ],
            "en-kk": [
                "dev/newsdev2019-enkk-src.en.sgm",
                "dev/newsdev2019-enkk-ref.kk.sgm",
            ],
        },
    ),
    "wmt19/google/ar": WMTAdditionDataset(
        "wmt19/google/ar",
        data=[
            "https://data.statmt.org/wmt19/translation-task/test.tgz",
            "https://raw.githubusercontent.com/google/wmt19-paraphrased-references/master/wmt19/ende/wmt19-ende-ar.ref",
        ],
        description="Additional high-quality reference for WMT19/en-de.",
        md5=["84de7162d158e28403103b01aeefc39a", "d66d9e91548ced0ac476f2390e32e2de"],
        citation="@misc{freitag2020bleu,\n    title={{BLEU} might be Guilty but References are not Innocent},\n    author={Markus Freitag and David Grangier and Isaac Caswell},\n    year={2020},\n    eprint={2004.06063},\n    archivePrefix={arXiv},\n    primaryClass={cs.CL}",
        langpairs={
            "en-de": ["sgm/newstest2019-ende-src.en.sgm", "wmt19_google_ar.wmt19-ende-ar.ref"],
        },
    ),
    "wmt19/google/arp": WMTAdditionDataset(
        "wmt19/google/arp",
        data=[
            "https://data.statmt.org/wmt19/translation-task/test.tgz",
            "https://raw.githubusercontent.com/google/wmt19-paraphrased-references/master/wmt19/ende/wmt19-ende-arp.ref",
        ],
        description="Additional paraphrase of wmt19/google/ar.",
        md5=["84de7162d158e28403103b01aeefc39a", "c70ea808cf2bff621ad7a8fddd4deca9"],
        citation="@misc{freitag2020bleu,\n    title={{BLEU} might be Guilty but References are not Innocent},\n    author={Markus Freitag and David Grangier and Isaac Caswell},\n    year={2020},\n    eprint={2004.06063},\n    archivePrefix={arXiv},\n    primaryClass={cs.CL}",
        langpairs={
            "en-de": ["sgm/newstest2019-ende-src.en.sgm", "wmt19_google_arp.wmt19-ende-arp.ref"],
        },
    ),
    "wmt19/google/wmtp": WMTAdditionDataset(
        "wmt19/google/wmtp",
        data=[
            "https://data.statmt.org/wmt19/translation-task/test.tgz",
            "https://raw.githubusercontent.com/google/wmt19-paraphrased-references/master/wmt19/ende/wmt19-ende-wmtp.ref",
        ],
        description="Additional paraphrase of the official WMT19 reference.",
        md5=["84de7162d158e28403103b01aeefc39a", "587c660ee5fd44727f0db025b71c6a82"],
        citation="@misc{freitag2020bleu,\n    title={{BLEU} might be Guilty but References are not Innocent},\n    author={Markus Freitag and David Grangier and Isaac Caswell},\n    year={2020},\n    eprint={2004.06063},\n    archivePrefix={arXiv},\n    primaryClass={cs.CL}",
        langpairs={
            "en-de": ["sgm/newstest2019-ende-src.en.sgm", "wmt19_google_wmtp.wmt19-ende-wmtp.ref"],
        },
    ),
    "wmt19/google/hqr": WMTAdditionDataset(
        "wmt19/google/hqr",
        data=[
            "https://data.statmt.org/wmt19/translation-task/test.tgz",
            "https://raw.githubusercontent.com/google/wmt19-paraphrased-references/master/wmt19/ende/wmt19-ende-hqr.ref",
        ],
        description="Best human selected-reference between wmt19 and wmt19/google/ar.",
        md5=["84de7162d158e28403103b01aeefc39a", "d9221135f62d7152de041f5bfc8efaea"],
        citation="@misc{freitag2020bleu,\n    title={{BLEU} might be Guilty but References are not Innocent},\n    author={Markus Freitag and David Grangier and Isaac Caswell},\n    year={2020},\n    eprint={2004.06063},\n    archivePrefix={arXiv},\n    primaryClass={cs.CL}",
        langpairs={
            "en-de": ["sgm/newstest2019-ende-src.en.sgm", "wmt19_google_hqr.wmt19-ende-hqr.ref"],
        },
    ),
    "wmt19/google/hqp": WMTAdditionDataset(
        "wmt19/google/hqp",
        data=[
            "https://data.statmt.org/wmt19/translation-task/test.tgz",
            "https://raw.githubusercontent.com/google/wmt19-paraphrased-references/master/wmt19/ende/wmt19-ende-hqp.ref",
        ],
        description="Best human-selected reference between wmt19/google/arp and wmt19/google/wmtp.",
        md5=["84de7162d158e28403103b01aeefc39a", "b7c3a07a59c8eccea5367e9ec5417a8a"],
        citation="@misc{freitag2020bleu,\n    title={{BLEU} might be Guilty but References are not Innocent},\n    author={Markus Freitag and David Grangier and Isaac Caswell},\n    year={2020},\n    eprint={2004.06063},\n    archivePrefix={arXiv},\n    primaryClass={cs.CL}",
        langpairs={
            "en-de": ["sgm/newstest2019-ende-src.en.sgm", "wmt19_google_hqp.wmt19-ende-hqp.ref"],
        },
    ),
    "wmt19/google/hqall": WMTAdditionDataset(
        "wmt19/google/hqall",
        data=[
            "https://data.statmt.org/wmt19/translation-task/test.tgz",
            "https://raw.githubusercontent.com/google/wmt19-paraphrased-references/master/wmt19/ende/wmt19-ende-hqall.ref",
        ],
        description="Best human-selected reference among original official reference and the Google reference and paraphrases.",
        md5=["84de7162d158e28403103b01aeefc39a", "edecf10ced59e10b703a6fbcf1fa9dfa"],
        citation="@misc{freitag2020bleu,\n    title={{BLEU} might be Guilty but References are not Innocent},\n    author={Markus Freitag and David Grangier and Isaac Caswell},\n    year={2020},\n    eprint={2004.06063},\n    archivePrefix={arXiv},\n    primaryClass={cs.CL}",
        langpairs={
            "en-de": ["sgm/newstest2019-ende-src.en.sgm", "wmt19_google_hqall.wmt19-ende-hqall.ref"],
        },
    ),
    "wmt18": FakeSGMLDataset(
        "wmt18",
        data=["https://data.statmt.org/wmt18/translation-task/test.tgz"],
        md5=["f996c245ecffea23d0006fa4c34e9064"],
        description="Official evaluation data.",
        citation='@inproceedings{bojar-etal-2018-findings,\n    title = "Findings of the 2018 Conference on Machine Translation ({WMT}18)",\n    author = "Bojar, Ond{\v{r}}ej  and\n      Federmann, Christian  and\n      Fishel, Mark  and\n      Graham, Yvette  and\n      Haddow, Barry  and\n      Koehn, Philipp  and\n      Monz, Christof",\n    booktitle = "Proceedings of the Third Conference on Machine Translation: Shared Task Papers",\n    month = oct,\n    year = "2018",\n    address = "Belgium, Brussels",\n    publisher = "Association for Computational Linguistics",\n    url = "https://www.aclweb.org/anthology/W18-6401",\n    pages = "272--303",\n}',
        langpairs={
            "cs-en": [
                "test/newstest2018-csen-src.cs.sgm",
                "test/newstest2018-csen-ref.en.sgm",
            ],
            "de-en": [
                "test/newstest2018-deen-src.de.sgm",
                "test/newstest2018-deen-ref.en.sgm",
            ],
            "en-cs": [
                "test/newstest2018-encs-src.en.sgm",
                "test/newstest2018-encs-ref.cs.sgm",
            ],
            "en-de": [
                "test/newstest2018-ende-src.en.sgm",
                "test/newstest2018-ende-ref.de.sgm",
            ],
            "en-et": [
                "test/newstest2018-enet-src.en.sgm",
                "test/newstest2018-enet-ref.et.sgm",
            ],
            "en-fi": [
                "test/newstest2018-enfi-src.en.sgm",
                "test/newstest2018-enfi-ref.fi.sgm",
            ],
            "en-ru": [
                "test/newstest2018-enru-src.en.sgm",
                "test/newstest2018-enru-ref.ru.sgm",
            ],
            "et-en": [
                "test/newstest2018-eten-src.et.sgm",
                "test/newstest2018-eten-ref.en.sgm",
            ],
            "fi-en": [
                "test/newstest2018-fien-src.fi.sgm",
                "test/newstest2018-fien-ref.en.sgm",
            ],
            "ru-en": [
                "test/newstest2018-ruen-src.ru.sgm",
                "test/newstest2018-ruen-ref.en.sgm",
            ],
            "en-tr": [
                "test/newstest2018-entr-src.en.sgm",
                "test/newstest2018-entr-ref.tr.sgm",
            ],
            "tr-en": [
                "test/newstest2018-tren-src.tr.sgm",
                "test/newstest2018-tren-ref.en.sgm",
            ],
            "en-zh": [
                "test/newstest2018-enzh-src.en.sgm",
                "test/newstest2018-enzh-ref.zh.sgm",
            ],
            "zh-en": [
                "test/newstest2018-zhen-src.zh.sgm",
                "test/newstest2018-zhen-ref.en.sgm",
            ],
        },
    ),
    "wmt18/test-ts": FakeSGMLDataset(
        "wmt18/test-ts",
        data=["https://data.statmt.org/wmt18/translation-task/test-ts.tgz"],
        md5=["5c621a34d512cc2dd74162ae7d00b320"],
        description="Official evaluation sources with extra test sets interleaved.",
        langpairs={
            "cs-en": ["test-ts/newstest2018-csen-src-ts.cs.sgm", "test-ts/newstest2018-csen-ref-ts.en.sgm"],
            "de-en": ["test-ts/newstest2018-deen-src-ts.de.sgm", "test-ts/newstest2018-deen-ref-ts.en.sgm"],
            "en-cs": ["test-ts/newstest2018-encs-src-ts.en.sgm", "test-ts/newstest2018-encs-ref-ts.cs.sgm"],
            "en-de": ["test-ts/newstest2018-ende-src-ts.en.sgm", "test-ts/newstest2018-ende-ref-ts.de.sgm"],
            "en-et": ["test-ts/newstest2018-enet-src-ts.en.sgm", "test-ts/newstest2018-enet-ref-ts.et.sgm"],
            "en-fi": ["test-ts/newstest2018-enfi-src-ts.en.sgm", "test-ts/newstest2018-enfi-ref-ts.fi.sgm"],
            "en-ru": ["test-ts/newstest2018-enru-src-ts.en.sgm", "test-ts/newstest2018-enru-ref-ts.ru.sgm"],
            "et-en": ["test-ts/newstest2018-eten-src-ts.et.sgm", "test-ts/newstest2018-eten-ref-ts.en.sgm"],
            "fi-en": ["test-ts/newstest2018-fien-src-ts.fi.sgm", "test-ts/newstest2018-fien-ref-ts.en.sgm"],
            "ru-en": ["test-ts/newstest2018-ruen-src-ts.ru.sgm", "test-ts/newstest2018-ruen-ref-ts.en.sgm"],
            "en-tr": ["test-ts/newstest2018-entr-src-ts.en.sgm", "test-ts/newstest2018-entr-ref-ts.tr.sgm"],
            "tr-en": ["test-ts/newstest2018-tren-src-ts.tr.sgm", "test-ts/newstest2018-tren-ref-ts.en.sgm"],
            "en-zh": ["test-ts/newstest2018-enzh-src-ts.en.sgm", "test-ts/newstest2018-enzh-ref-ts.zh.sgm"],
            "zh-en": ["test-ts/newstest2018-zhen-src-ts.zh.sgm", "test-ts/newstest2018-zhen-ref-ts.en.sgm"],
        },
    ),
    "wmt18/dev": FakeSGMLDataset(
        "wmt18/dev",
        data=["https://data.statmt.org/wmt18/translation-task/dev.tgz"],
        md5=["486f391da54a7a3247f02ebd25996f24"],
        description="Development data (Estonian<>English).",
        langpairs={
            "et-en": [
                "dev/newsdev2018-eten-src.et.sgm",
                "dev/newsdev2018-eten-ref.en.sgm",
            ],
            "en-et": [
                "dev/newsdev2018-enet-src.en.sgm",
                "dev/newsdev2018-enet-ref.et.sgm",
            ],
        },
    ),
    "wmt17": FakeSGMLDataset(
        "wmt17",
        data=["https://data.statmt.org/wmt17/translation-task/test.tgz"],
        md5=["86a1724c276004aa25455ae2a04cef26"],
        description="Official evaluation data.",
        citation="@InProceedings{bojar-EtAl:2017:WMT1,\n  author    = {Bojar, Ond\\v{r}ej  and  Chatterjee, Rajen  and  Federmann, Christian  and  Graham, Yvette  and  Haddow, Barry  and  Huang, Shujian  and  Huck, Matthias  and  Koehn, Philipp  and  Liu, Qun  and  Logacheva, Varvara  and  Monz, Christof  and  Negri, Matteo  and  Post, Matt  and  Rubino, Raphael  and  Specia, Lucia  and  Turchi, Marco},\n  title     = {Findings of the 2017 Conference on Machine Translation (WMT17)},\n  booktitle = {Proceedings of the Second Conference on Machine Translation, Volume 2: Shared Task Papers},\n  month     = {September},\n  year      = {2017},\n  address   = {Copenhagen, Denmark},\n  publisher = {Association for Computational Linguistics},\n  pages     = {169--214},\n  url       = {http://www.aclweb.org/anthology/W17-4717}\n}",
        langpairs={
            "cs-en": [
                "test/newstest2017-csen-src.cs.sgm",
                "test/newstest2017-csen-ref.en.sgm",
            ],
            "de-en": [
                "test/newstest2017-deen-src.de.sgm",
                "test/newstest2017-deen-ref.en.sgm",
            ],
            "en-cs": [
                "test/newstest2017-encs-src.en.sgm",
                "test/newstest2017-encs-ref.cs.sgm",
            ],
            "en-de": [
                "test/newstest2017-ende-src.en.sgm",
                "test/newstest2017-ende-ref.de.sgm",
            ],
            "en-fi": [
                "test/newstest2017-enfi-src.en.sgm",
                "test/newstest2017-enfi-ref.fi.sgm",
            ],
            "en-lv": [
                "test/newstest2017-enlv-src.en.sgm",
                "test/newstest2017-enlv-ref.lv.sgm",
            ],
            "en-ru": [
                "test/newstest2017-enru-src.en.sgm",
                "test/newstest2017-enru-ref.ru.sgm",
            ],
            "en-tr": [
                "test/newstest2017-entr-src.en.sgm",
                "test/newstest2017-entr-ref.tr.sgm",
            ],
            "en-zh": [
                "test/newstest2017-enzh-src.en.sgm",
                "test/newstest2017-enzh-ref.zh.sgm",
            ],
            "fi-en": [
                "test/newstest2017-fien-src.fi.sgm",
                "test/newstest2017-fien-ref.en.sgm",
            ],
            "lv-en": [
                "test/newstest2017-lven-src.lv.sgm",
                "test/newstest2017-lven-ref.en.sgm",
            ],
            "ru-en": [
                "test/newstest2017-ruen-src.ru.sgm",
                "test/newstest2017-ruen-ref.en.sgm",
            ],
            "tr-en": [
                "test/newstest2017-tren-src.tr.sgm",
                "test/newstest2017-tren-ref.en.sgm",
            ],
            "zh-en": [
                "test/newstest2017-zhen-src.zh.sgm",
                "test/newstest2017-zhen-ref.en.sgm",
            ],
        },
    ),
    "wmt17/B": FakeSGMLDataset(
        "wmt17/B",
        data=["https://data.statmt.org/wmt17/translation-task/test.tgz"],
        md5=["86a1724c276004aa25455ae2a04cef26"],
        description="Additional reference for EN-FI and FI-EN.",
        langpairs={
            "en-fi": [
                "test/newstestB2017-enfi-src.en.sgm",
                "test/newstestB2017-enfi-ref.fi.sgm",
            ],
        },
    ),
    "wmt17/tworefs": FakeSGMLDataset(
        "wmt17/tworefs",
        data=["https://data.statmt.org/wmt17/translation-task/test.tgz"],
        md5=["86a1724c276004aa25455ae2a04cef26"],
        description="Systems with two references.",
        langpairs={
            "en-fi": [
                "test/newstest2017-enfi-src.en.sgm",
                "test/newstest2017-enfi-ref.fi.sgm",
                "test/newstestB2017-enfi-ref.fi.sgm",
            ],
        },
    ),
    "wmt17/improved": FakeSGMLDataset(
        "wmt17/improved",
        data=["https://data.statmt.org/wmt17/translation-task/test-update-1.tgz"],
        md5=["91dbfd5af99bc6891a637a68e04dfd41"],
        description="Improved zh-en and en-zh translations.",
        langpairs={
            "en-zh": ["newstest2017-enzh-src.en.sgm", "newstest2017-enzh-ref.zh.sgm"],
            "zh-en": ["newstest2017-zhen-src.zh.sgm", "newstest2017-zhen-ref.en.sgm"],
        },
    ),
    "wmt17/dev": FakeSGMLDataset(
        "wmt17/dev",
        data=["https://data.statmt.org/wmt17/translation-task/dev.tgz"],
        md5=["9b1aa63c1cf49dccdd20b962fe313989"],
        description="Development sets released for new languages in 2017.",
        langpairs={
            "en-lv": [
                "dev/newsdev2017-enlv-src.en.sgm",
                "dev/newsdev2017-enlv-ref.lv.sgm",
            ],
            "en-zh": [
                "dev/newsdev2017-enzh-src.en.sgm",
                "dev/newsdev2017-enzh-ref.zh.sgm",
            ],
            "lv-en": [
                "dev/newsdev2017-lven-src.lv.sgm",
                "dev/newsdev2017-lven-ref.en.sgm",
            ],
            "zh-en": [
                "dev/newsdev2017-zhen-src.zh.sgm",
                "dev/newsdev2017-zhen-ref.en.sgm",
            ],
        },
    ),
    "wmt17/ms": WMTAdditionDataset(
        "wmt17/ms",
        data=[
            "https://github.com/MicrosoftTranslator/Translator-HumanParityData/archive/master.zip",
            "https://data.statmt.org/wmt17/translation-task/test-update-1.tgz",
        ],
        md5=["18fdaa7a3c84cf6ef688da1f6a5fa96f", "91dbfd5af99bc6891a637a68e04dfd41"],
        description="Additional Chinese-English references from Microsoft Research.",
        citation="@inproceedings{achieving-human-parity-on-automatic-chinese-to-english-news-translation,\n  author = {Hassan Awadalla, Hany and Aue, Anthony and Chen, Chang and Chowdhary, Vishal and Clark, Jonathan and Federmann, Christian and Huang, Xuedong and Junczys-Dowmunt, Marcin and Lewis, Will and Li, Mu and Liu, Shujie and Liu, Tie-Yan and Luo, Renqian and Menezes, Arul and Qin, Tao and Seide, Frank and Tan, Xu and Tian, Fei and Wu, Lijun and Wu, Shuangzhi and Xia, Yingce and Zhang, Dongdong and Zhang, Zhirui and Zhou, Ming},\n  title = {Achieving Human Parity on Automatic Chinese to English News Translation},\n  booktitle = {},\n  year = {2018},\n  month = {March},\n  abstract = {Machine translation has made rapid advances in recent years. Millions of people are using it today in online translation systems and mobile applications in order to communicate across language barriers. The question naturally arises whether such systems can approach or achieve parity with human translations. In this paper, we first address the problem of how to define and accurately measure human parity in translation. We then describe Microsoft’s machine translation system and measure the quality of its translations on the widely used WMT 2017 news translation task from Chinese to English. We find that our latest neural machine translation system has reached a new state-of-the-art, and that the translation quality is at human parity when compared to professional human translations. We also find that it significantly exceeds the quality of crowd-sourced non-professional translations.},\n  publisher = {},\n  url = {https://www.microsoft.com/en-us/research/publication/achieving-human-parity-on-automatic-chinese-to-english-news-translation/},\n  address = {},\n  pages = {},\n  journal = {},\n  volume = {},\n  chapter = {},\n  isbn = {},\n}",
        langpairs={
            "zh-en": [
                "newstest2017-zhen-src.zh.sgm",
                "newstest2017-zhen-ref.en.sgm",
                "Translator-HumanParityData-master/Translator-HumanParityData/References/Translator-HumanParityData-Reference-HT.txt",
                "Translator-HumanParityData-master/Translator-HumanParityData/References/Translator-HumanParityData-Reference-PE.txt",
            ],
        },
    ),
    "wmt16": FakeSGMLDataset(
        "wmt16",
        data=["https://data.statmt.org/wmt16/translation-task/test.tgz"],
        md5=["3d809cd0c2c86adb2c67034d15c4e446"],
        description="Official evaluation data.",
        citation="@InProceedings{bojar-EtAl:2016:WMT1,\n  author    = {Bojar, Ond\\v{r}ej  and  Chatterjee, Rajen  and  Federmann, Christian  and  Graham, Yvette  and  Haddow, Barry  and  Huck, Matthias  and  Jimeno Yepes, Antonio  and  Koehn, Philipp  and  Logacheva, Varvara  and  Monz, Christof  and  Negri, Matteo  and  Neveol, Aurelie  and  Neves, Mariana  and  Popel, Martin  and  Post, Matt  and  Rubino, Raphael  and  Scarton, Carolina  and  Specia, Lucia  and  Turchi, Marco  and  Verspoor, Karin  and  Zampieri, Marcos},\n  title     = {Findings of the 2016 Conference on Machine Translation},\n  booktitle = {Proceedings of the First Conference on Machine Translation},\n  month     = {August},\n  year      = {2016},\n  address   = {Berlin, Germany},\n  publisher = {Association for Computational Linguistics},\n  pages     = {131--198},\n  url       = {http://www.aclweb.org/anthology/W/W16/W16-2301}\n}",
        langpairs={
            "cs-en": [
                "test/newstest2016-csen-src.cs.sgm",
                "test/newstest2016-csen-ref.en.sgm",
            ],
            "de-en": [
                "test/newstest2016-deen-src.de.sgm",
                "test/newstest2016-deen-ref.en.sgm",
            ],
            "en-cs": [
                "test/newstest2016-encs-src.en.sgm",
                "test/newstest2016-encs-ref.cs.sgm",
            ],
            "en-de": [
                "test/newstest2016-ende-src.en.sgm",
                "test/newstest2016-ende-ref.de.sgm",
            ],
            "en-fi": [
                "test/newstest2016-enfi-src.en.sgm",
                "test/newstest2016-enfi-ref.fi.sgm",
            ],
            "en-ro": [
                "test/newstest2016-enro-src.en.sgm",
                "test/newstest2016-enro-ref.ro.sgm",
            ],
            "en-ru": [
                "test/newstest2016-enru-src.en.sgm",
                "test/newstest2016-enru-ref.ru.sgm",
            ],
            "en-tr": [
                "test/newstest2016-entr-src.en.sgm",
                "test/newstest2016-entr-ref.tr.sgm",
            ],
            "fi-en": [
                "test/newstest2016-fien-src.fi.sgm",
                "test/newstest2016-fien-ref.en.sgm",
            ],
            "ro-en": [
                "test/newstest2016-roen-src.ro.sgm",
                "test/newstest2016-roen-ref.en.sgm",
            ],
            "ru-en": [
                "test/newstest2016-ruen-src.ru.sgm",
                "test/newstest2016-ruen-ref.en.sgm",
            ],
            "tr-en": [
                "test/newstest2016-tren-src.tr.sgm",
                "test/newstest2016-tren-ref.en.sgm",
            ],
        },
    ),
    "wmt16/B": FakeSGMLDataset(
        "wmt16/B",
        data=["https://data.statmt.org/wmt16/translation-task/test.tgz"],
        md5=["3d809cd0c2c86adb2c67034d15c4e446"],
        description="Additional reference for EN-FI.",
        langpairs={
            "en-fi": [
                "test/newstest2016-enfi-src.en.sgm",
                "test/newstestB2016-enfi-ref.fi.sgm",
            ],
        },
    ),
    "wmt16/tworefs": FakeSGMLDataset(
        "wmt16/tworefs",
        data=["https://data.statmt.org/wmt16/translation-task/test.tgz"],
        md5=["3d809cd0c2c86adb2c67034d15c4e446"],
        description="EN-FI with two references.",
        langpairs={
            "en-fi": [
                "test/newstest2016-enfi-src.en.sgm",
                "test/newstest2016-enfi-ref.fi.sgm",
                "test/newstestB2016-enfi-ref.fi.sgm",
            ],
        },
    ),
    "wmt16/dev": FakeSGMLDataset(
        "wmt16/dev",
        data=["https://data.statmt.org/wmt16/translation-task/dev.tgz"],
        md5=["4a3dc2760bb077f4308cce96b06e6af6"],
        description="Development sets released for new languages in 2016.",
        langpairs={
            "en-ro": [
                "dev/newsdev2016-enro-src.en.sgm",
                "dev/newsdev2016-enro-ref.ro.sgm",
            ],
            "en-tr": [
                "dev/newsdev2016-entr-src.en.sgm",
                "dev/newsdev2016-entr-ref.tr.sgm",
            ],
            "ro-en": [
                "dev/newsdev2016-roen-src.ro.sgm",
                "dev/newsdev2016-roen-ref.en.sgm",
            ],
            "tr-en": [
                "dev/newsdev2016-tren-src.tr.sgm",
                "dev/newsdev2016-tren-ref.en.sgm",
            ],
        },
    ),
    "wmt15": FakeSGMLDataset(
        "wmt15",
        data=["https://statmt.org/wmt15/test.tgz"],
        md5=["67e3beca15e69fe3d36de149da0a96df"],
        description="Official evaluation data.",
        citation="@InProceedings{bojar-EtAl:2015:WMT,\n  author    = {Bojar, Ond\\v{r}ej  and  Chatterjee, Rajen  and  Federmann, Christian  and  Haddow, Barry  and  Huck, Matthias  and  Hokamp, Chris  and  Koehn, Philipp  and  Logacheva, Varvara  and  Monz, Christof  and  Negri, Matteo  and  Post, Matt  and  Scarton, Carolina  and  Specia, Lucia  and  Turchi, Marco},\n  title     = {Findings of the 2015 Workshop on Statistical Machine Translation},\n  booktitle = {Proceedings of the Tenth Workshop on Statistical Machine Translation},\n  month     = {September},\n  year      = {2015},\n  address   = {Lisbon, Portugal},\n  publisher = {Association for Computational Linguistics},\n  pages     = {1--46},\n  url       = {http://aclweb.org/anthology/W15-3001}\n}",
        langpairs={
            "en-fr": [
                "test/newsdiscusstest2015-enfr-src.en.sgm",
                "test/newsdiscusstest2015-enfr-ref.fr.sgm",
            ],
            "fr-en": [
                "test/newsdiscusstest2015-fren-src.fr.sgm",
                "test/newsdiscusstest2015-fren-ref.en.sgm",
            ],
            "cs-en": [
                "test/newstest2015-csen-src.cs.sgm",
                "test/newstest2015-csen-ref.en.sgm",
            ],
            "de-en": [
                "test/newstest2015-deen-src.de.sgm",
                "test/newstest2015-deen-ref.en.sgm",
            ],
            "en-cs": [
                "test/newstest2015-encs-src.en.sgm",
                "test/newstest2015-encs-ref.cs.sgm",
            ],
            "en-de": [
                "test/newstest2015-ende-src.en.sgm",
                "test/newstest2015-ende-ref.de.sgm",
            ],
            "en-fi": [
                "test/newstest2015-enfi-src.en.sgm",
                "test/newstest2015-enfi-ref.fi.sgm",
            ],
            "en-ru": [
                "test/newstest2015-enru-src.en.sgm",
                "test/newstest2015-enru-ref.ru.sgm",
            ],
            "fi-en": [
                "test/newstest2015-fien-src.fi.sgm",
                "test/newstest2015-fien-ref.en.sgm",
            ],
            "ru-en": [
                "test/newstest2015-ruen-src.ru.sgm",
                "test/newstest2015-ruen-ref.en.sgm",
            ],
        },
    ),
    "wmt14": FakeSGMLDataset(
        "wmt14",
        data=["https://statmt.org/wmt14/test-filtered.tgz"],
        md5=["84c597844c1542e29c2aff23aaee4310"],
        description="Official evaluation data.",
        citation="@InProceedings{bojar-EtAl:2014:W14-33,\n  author    = {Bojar, Ondrej  and  Buck, Christian  and  Federmann, Christian  and  Haddow, Barry  and  Koehn, Philipp  and  Leveling, Johannes  and  Monz, Christof  and  Pecina, Pavel  and  Post, Matt  and  Saint-Amand, Herve  and  Soricut, Radu  and  Specia, Lucia  and  Tamchyna, Ale\\v{s}},\n  title     = {Findings of the 2014 Workshop on Statistical Machine Translation},\n  booktitle = {Proceedings of the Ninth Workshop on Statistical Machine Translation},\n  month     = {June},\n  year      = {2014},\n  address   = {Baltimore, Maryland, USA},\n  publisher = {Association for Computational Linguistics},\n  pages     = {12--58},\n  url       = {http://www.aclweb.org/anthology/W/W14/W14-3302}\n}",
        langpairs={
            "cs-en": [
                "test/newstest2014-csen-src.cs.sgm",
                "test/newstest2014-csen-ref.en.sgm",
            ],
            "en-cs": [
                "test/newstest2014-csen-src.en.sgm",
                "test/newstest2014-csen-ref.cs.sgm",
            ],
            "de-en": [
                "test/newstest2014-deen-src.de.sgm",
                "test/newstest2014-deen-ref.en.sgm",
            ],
            "en-de": [
                "test/newstest2014-deen-src.en.sgm",
                "test/newstest2014-deen-ref.de.sgm",
            ],
            "en-fr": [
                "test/newstest2014-fren-src.en.sgm",
                "test/newstest2014-fren-ref.fr.sgm",
            ],
            "fr-en": [
                "test/newstest2014-fren-src.fr.sgm",
                "test/newstest2014-fren-ref.en.sgm",
            ],
            "en-hi": [
                "test/newstest2014-hien-src.en.sgm",
                "test/newstest2014-hien-ref.hi.sgm",
            ],
            "hi-en": [
                "test/newstest2014-hien-src.hi.sgm",
                "test/newstest2014-hien-ref.en.sgm",
            ],
            "en-ru": [
                "test/newstest2014-ruen-src.en.sgm",
                "test/newstest2014-ruen-ref.ru.sgm",
            ],
            "ru-en": [
                "test/newstest2014-ruen-src.ru.sgm",
                "test/newstest2014-ruen-ref.en.sgm",
            ],
        },
    ),
    "wmt14/full": FakeSGMLDataset(
        "wmt14/full",
        data=["https://statmt.org/wmt14/test-full.tgz"],
        md5=["a8cd784e006feb32ac6f3d9ec7eb389a"],
        description="Evaluation data released after official evaluation for further research.",
        langpairs={
            "cs-en": [
                "test-full/newstest2014-csen-src.cs.sgm",
                "test-full/newstest2014-csen-ref.en.sgm",
            ],
            "en-cs": [
                "test-full/newstest2014-csen-src.en.sgm",
                "test-full/newstest2014-csen-ref.cs.sgm",
            ],
            "de-en": [
                "test-full/newstest2014-deen-src.de.sgm",
                "test-full/newstest2014-deen-ref.en.sgm",
            ],
            "en-de": [
                "test-full/newstest2014-deen-src.en.sgm",
                "test-full/newstest2014-deen-ref.de.sgm",
            ],
            "en-fr": [
                "test-full/newstest2014-fren-src.en.sgm",
                "test-full/newstest2014-fren-ref.fr.sgm",
            ],
            "fr-en": [
                "test-full/newstest2014-fren-src.fr.sgm",
                "test-full/newstest2014-fren-ref.en.sgm",
            ],
            "en-hi": [
                "test-full/newstest2014-hien-src.en.sgm",
                "test-full/newstest2014-hien-ref.hi.sgm",
            ],
            "hi-en": [
                "test-full/newstest2014-hien-src.hi.sgm",
                "test-full/newstest2014-hien-ref.en.sgm",
            ],
            "en-ru": [
                "test-full/newstest2014-ruen-src.en.sgm",
                "test-full/newstest2014-ruen-ref.ru.sgm",
            ],
            "ru-en": [
                "test-full/newstest2014-ruen-src.ru.sgm",
                "test-full/newstest2014-ruen-ref.en.sgm",
            ],
        },
    ),
    "wmt13": FakeSGMLDataset(
        "wmt13",
        data=["https://statmt.org/wmt13/test.tgz"],
        md5=["48eca5d02f637af44e85186847141f67"],
        description="Official evaluation data.",
        citation="@InProceedings{bojar-EtAl:2013:WMT,\n  author    = {Bojar, Ond\\v{r}ej  and  Buck, Christian  and  Callison-Burch, Chris  and  Federmann, Christian  and  Haddow, Barry  and  Koehn, Philipp  and  Monz, Christof  and  Post, Matt  and  Soricut, Radu  and  Specia, Lucia},\n  title     = {Findings of the 2013 {Workshop on Statistical Machine Translation}},\n  booktitle = {Proceedings of the Eighth Workshop on Statistical Machine Translation},\n  month     = {August},\n  year      = {2013},\n  address   = {Sofia, Bulgaria},\n  publisher = {Association for Computational Linguistics},\n  pages     = {1--44},\n  url       = {http://www.aclweb.org/anthology/W13-2201}\n}",
        langpairs={
            "cs-en": ["test/newstest2013-src.cs.sgm", "test/newstest2013-src.en.sgm"],
            "en-cs": ["test/newstest2013-src.en.sgm", "test/newstest2013-src.cs.sgm"],
            "de-en": ["test/newstest2013-src.de.sgm", "test/newstest2013-src.en.sgm"],
            "en-de": ["test/newstest2013-src.en.sgm", "test/newstest2013-src.de.sgm"],
            "es-en": ["test/newstest2013-src.es.sgm", "test/newstest2013-src.en.sgm"],
            "en-es": ["test/newstest2013-src.en.sgm", "test/newstest2013-src.es.sgm"],
            "fr-en": ["test/newstest2013-src.fr.sgm", "test/newstest2013-src.en.sgm"],
            "en-fr": ["test/newstest2013-src.en.sgm", "test/newstest2013-src.fr.sgm"],
            "ru-en": ["test/newstest2013-src.ru.sgm", "test/newstest2013-src.en.sgm"],
            "en-ru": ["test/newstest2013-src.en.sgm", "test/newstest2013-src.ru.sgm"],
        },
    ),
    "wmt12": FakeSGMLDataset(
        "wmt12",
        data=["https://statmt.org/wmt12/test.tgz"],
        md5=["608232d34ebc4ba2ff70fead45674e47"],
        description="Official evaluation data.",
        citation="@InProceedings{callisonburch-EtAl:2012:WMT,\n  author    = {Callison-Burch, Chris  and  Koehn, Philipp  and  Monz, Christof  and  Post, Matt  and  Soricut, Radu  and  Specia, Lucia},\n  title     = {Findings of the 2012 Workshop on Statistical Machine Translation},\n  booktitle = {Proceedings of the Seventh Workshop on Statistical Machine Translation},\n  month     = {June},\n  year      = {2012},\n  address   = {Montr{'e}al, Canada},\n  publisher = {Association for Computational Linguistics},\n  pages     = {10--51},\n  url       = {http://www.aclweb.org/anthology/W12-3102}\n}",
        langpairs={
            "cs-en": ["test/newstest2012-src.cs.sgm", "test/newstest2012-src.en.sgm"],
            "en-cs": ["test/newstest2012-src.en.sgm", "test/newstest2012-src.cs.sgm"],
            "de-en": ["test/newstest2012-src.de.sgm", "test/newstest2012-src.en.sgm"],
            "en-de": ["test/newstest2012-src.en.sgm", "test/newstest2012-src.de.sgm"],
            "es-en": ["test/newstest2012-src.es.sgm", "test/newstest2012-src.en.sgm"],
            "en-es": ["test/newstest2012-src.en.sgm", "test/newstest2012-src.es.sgm"],
            "fr-en": ["test/newstest2012-src.fr.sgm", "test/newstest2012-src.en.sgm"],
            "en-fr": ["test/newstest2012-src.en.sgm", "test/newstest2012-src.fr.sgm"],
        },
    ),
    "wmt11": FakeSGMLDataset(
        "wmt11",
        data=["https://statmt.org/wmt11/test.tgz"],
        md5=["b0c9680adf32d394aefc2b24e3a5937e"],
        description="Official evaluation data.",
        citation="@InProceedings{callisonburch-EtAl:2011:WMT,\n  author    = {Callison-Burch, Chris  and  Koehn, Philipp  and  Monz, Christof  and  Zaidan, Omar},\n  title     = {Findings of the 2011 Workshop on Statistical Machine Translation},\n  booktitle = {Proceedings of the Sixth Workshop on Statistical Machine Translation},\n  month     = {July},\n  year      = {2011},\n  address   = {Edinburgh, Scotland},\n  publisher = {Association for Computational Linguistics},\n  pages     = {22--64},\n  url       = {http://www.aclweb.org/anthology/W11-2103}\n}",
        langpairs={
            "cs-en": ["newstest2011-src.cs.sgm", "newstest2011-src.en.sgm"],
            "en-cs": ["newstest2011-src.en.sgm", "newstest2011-src.cs.sgm"],
            "de-en": ["newstest2011-src.de.sgm", "newstest2011-src.en.sgm"],
            "en-de": ["newstest2011-src.en.sgm", "newstest2011-src.de.sgm"],
            "fr-en": ["newstest2011-src.fr.sgm", "newstest2011-src.en.sgm"],
            "en-fr": ["newstest2011-src.en.sgm", "newstest2011-src.fr.sgm"],
            "es-en": ["newstest2011-src.es.sgm", "newstest2011-src.en.sgm"],
            "en-es": ["newstest2011-src.en.sgm", "newstest2011-src.es.sgm"],
        },
    ),
    "wmt10": FakeSGMLDataset(
        "wmt10",
        data=["https://statmt.org/wmt10/test.tgz"],
        md5=["491cb885a355da5a23ea66e7b3024d5c"],
        description="Official evaluation data.",
        citation="@InProceedings{callisonburch-EtAl:2010:WMT,\n  author    = {Callison-Burch, Chris  and  Koehn, Philipp  and  Monz, Christof  and  Peterson, Kay  and  Przybocki, Mark  and  Zaidan, Omar},\n  title     = {Findings of the 2010 Joint Workshop on Statistical Machine Translation and Metrics for Machine Translation},\n  booktitle = {Proceedings of the Joint Fifth Workshop on Statistical Machine Translation and MetricsMATR},\n  month     = {July},\n  year      = {2010},\n  address   = {Uppsala, Sweden},\n  publisher = {Association for Computational Linguistics},\n  pages     = {17--53},\n  note      = {Revised August 2010},\n  url       = {http://www.aclweb.org/anthology/W10-1703}\n}",
        langpairs={
            "cs-en": ["test/newstest2010-src.cz.sgm", "test/newstest2010-src.en.sgm"],
            "en-cs": ["test/newstest2010-src.en.sgm", "test/newstest2010-src.cz.sgm"],
            "de-en": ["test/newstest2010-src.de.sgm", "test/newstest2010-src.en.sgm"],
            "en-de": ["test/newstest2010-src.en.sgm", "test/newstest2010-src.de.sgm"],
            "es-en": ["test/newstest2010-src.es.sgm", "test/newstest2010-src.en.sgm"],
            "en-es": ["test/newstest2010-src.en.sgm", "test/newstest2010-src.es.sgm"],
            "fr-en": ["test/newstest2010-src.fr.sgm", "test/newstest2010-src.en.sgm"],
            "en-fr": ["test/newstest2010-src.en.sgm", "test/newstest2010-src.fr.sgm"],
        },
    ),
    "wmt09": FakeSGMLDataset(
        "wmt09",
        data=["https://statmt.org/wmt09/test.tgz"],
        md5=["da227abfbd7b666ec175b742a0d27b37"],
        description="Official evaluation data.",
        citation="@InProceedings{callisonburch-EtAl:2009:WMT-09,\n  author    = {Callison-Burch, Chris  and  Koehn, Philipp  and  Monz, Christof  and  Schroeder, Josh},\n  title     = {Findings of the 2009 {W}orkshop on {S}tatistical {M}achine {T}ranslation},\n  booktitle = {Proceedings of the Fourth Workshop on Statistical Machine Translation},\n  month     = {March},\n  year      = {2009},\n  address   = {Athens, Greece},\n  publisher = {Association for Computational Linguistics},\n  pages     = {1--28},\n  url       = {http://www.aclweb.org/anthology/W/W09/W09-0401}\n}",
        langpairs={
            "cs-en": ["test/newstest2009-src.cz.sgm", "test/newstest2009-src.en.sgm"],
            "en-cs": ["test/newstest2009-src.en.sgm", "test/newstest2009-src.cz.sgm"],
            "de-en": ["test/newstest2009-src.de.sgm", "test/newstest2009-src.en.sgm"],
            "en-de": ["test/newstest2009-src.en.sgm", "test/newstest2009-src.de.sgm"],
            "es-en": ["test/newstest2009-src.es.sgm", "test/newstest2009-src.en.sgm"],
            "en-es": ["test/newstest2009-src.en.sgm", "test/newstest2009-src.es.sgm"],
            "fr-en": ["test/newstest2009-src.fr.sgm", "test/newstest2009-src.en.sgm"],
            "en-fr": ["test/newstest2009-src.en.sgm", "test/newstest2009-src.fr.sgm"],
            "hu-en": ["test/newstest2009-src.hu.sgm", "test/newstest2009-src.en.sgm"],
            "en-hu": ["test/newstest2009-src.en.sgm", "test/newstest2009-src.hu.sgm"],
            "it-en": ["test/newstest2009-src.it.sgm", "test/newstest2009-src.en.sgm"],
            "en-it": ["test/newstest2009-src.en.sgm", "test/newstest2009-src.it.sgm"],
        },
    ),
    "wmt08": FakeSGMLDataset(
        "wmt08",
        data=["https://statmt.org/wmt08/test.tgz"],
        md5=["0582e4e894a3342044059c894e1aea3d"],
        description="Official evaluation data.",
        citation="@InProceedings{callisonburch-EtAl:2008:WMT,\n  author    = {Callison-Burch, Chris  and  Fordyce, Cameron  and  Koehn, Philipp  and  Monz, Christof  and  Schroeder, Josh},\n  title     = {Further Meta-Evaluation of Machine Translation},\n  booktitle = {Proceedings of the Third Workshop on Statistical Machine Translation},\n  month     = {June},\n  year      = {2008},\n  address   = {Columbus, Ohio},\n  publisher = {Association for Computational Linguistics},\n  pages     = {70--106},\n  url       = {http://www.aclweb.org/anthology/W/W08/W08-0309}\n}",
        langpairs={
            "cs-en": ["test/newstest2008-src.cz.sgm", "test/newstest2008-src.en.sgm"],
            "en-cs": ["test/newstest2008-src.en.sgm", "test/newstest2008-src.cz.sgm"],
            "de-en": ["test/newstest2008-src.de.sgm", "test/newstest2008-src.en.sgm"],
            "en-de": ["test/newstest2008-src.en.sgm", "test/newstest2008-src.de.sgm"],
            "es-en": ["test/newstest2008-src.es.sgm", "test/newstest2008-src.en.sgm"],
            "en-es": ["test/newstest2008-src.en.sgm", "test/newstest2008-src.es.sgm"],
            "fr-en": ["test/newstest2008-src.fr.sgm", "test/newstest2008-src.en.sgm"],
            "en-fr": ["test/newstest2008-src.en.sgm", "test/newstest2008-src.fr.sgm"],
            "hu-en": ["test/newstest2008-src.hu.sgm", "test/newstest2008-src.en.sgm"],
            "en-hu": ["test/newstest2008-src.en.sgm", "test/newstest2008-src.hu.sgm"],
        },
    ),
    "wmt08/nc": FakeSGMLDataset(
        "wmt08/nc",
        data=["https://statmt.org/wmt08/test.tgz"],
        md5=["0582e4e894a3342044059c894e1aea3d"],
        description="Official evaluation data (news commentary).",
        langpairs={
            "cs-en": ["test/nc-test2008-src.cz.sgm", "test/nc-test2008-src.en.sgm"],
            "en-cs": ["test/nc-test2008-src.en.sgm", "test/nc-test2008-src.cz.sgm"],
        },
    ),
    "wmt08/europarl": FakeSGMLDataset(
        "wmt08/europarl",
        data=["https://statmt.org/wmt08/test.tgz"],
        md5=["0582e4e894a3342044059c894e1aea3d"],
        description="Official evaluation data (Europarl).",
        langpairs={
            "de-en": ["test/test2008-src.de.sgm", "test/test2008-src.en.sgm"],
            "en-de": ["test/test2008-src.en.sgm", "test/test2008-src.de.sgm"],
            "es-en": ["test/test2008-src.es.sgm", "test/test2008-src.en.sgm"],
            "en-es": ["test/test2008-src.en.sgm", "test/test2008-src.es.sgm"],
            "fr-en": ["test/test2008-src.fr.sgm", "test/test2008-src.en.sgm"],
            "en-fr": ["test/test2008-src.en.sgm", "test/test2008-src.fr.sgm"],
        },
    ),
    # iwslt
    "iwslt17": IWSLTXMLDataset(
        "iwslt17",
        data=[
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/en/fr/en-fr.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/fr/en/fr-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/en/de/en-de.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/de/en/de-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/en/ar/en-ar.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/ar/en/ar-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/en/ja/en-ja.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/ja/en/ja-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/en/ko/en-ko.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/ko/en/ko-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/en/zh/en-zh.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/zh/en/zh-en.tgz",
        ],
        md5=[
            "1849bcc3b006dc0642a8843b11aa7192",
            "79bf7a2ef02d226875f55fb076e7e473",
            "b68e7097b179491f6c466ef41ad72b9b",
            "e3f5b2a075a2da1a395c8b60bf1e9be1",
            "ecdc6bc4ab4c8984e919444f3c05183a",
            "4b5141d14b98706c081371e2f8afe0ca",
            "d957ee79de1f33c89077d37c5a2c5b06",
            "c213e8bb918ebf843543fe9fd2e33db2",
            "59f6a81c707378176e9ad8bb8d811f5f",
            "7e580af973bb389ec1d1378a1850742f",
            "975a858783a0ebec8c57d83ddd5bd381",
            "cc51d9b7fe1ff2af858c6a0dd80b8815",
        ],
        description="Official evaluation data for IWSLT.",
        citation="@InProceedings{iwslt2017,\n  author    = {Cettolo, Mauro and Federico, Marcello and Bentivogli, Luisa and Niehues, Jan and Stüker, Sebastian and Sudoh, Katsuitho and Yoshino, Koichiro and Federmann, Christian},\n  title     = {Overview of the IWSLT 2017 Evaluation Campaign},\n  booktitle = {14th International Workshop on Spoken Language Translation},\n  month     = {December},\n  year      = {2017},\n  address   = {Tokyo, Japan},\n  pages     = {2--14},\n  url       = {http://workshop2017.iwslt.org/downloads/iwslt2017_proceeding_v2.pdf}\n}",
        langpairs={
            "en-fr": [
                "en-fr/IWSLT17.TED.tst2017.en-fr.en.xml",
                "fr-en/IWSLT17.TED.tst2017.fr-en.fr.xml",
            ],
            "fr-en": [
                "fr-en/IWSLT17.TED.tst2017.fr-en.fr.xml",
                "en-fr/IWSLT17.TED.tst2017.en-fr.en.xml",
            ],
            "en-de": [
                "en-de/IWSLT17.TED.tst2017.en-de.en.xml",
                "de-en/IWSLT17.TED.tst2017.de-en.de.xml",
            ],
            "de-en": [
                "de-en/IWSLT17.TED.tst2017.de-en.de.xml",
                "en-de/IWSLT17.TED.tst2017.en-de.en.xml",
            ],
            "en-zh": [
                "en-zh/IWSLT17.TED.tst2017.en-zh.en.xml",
                "zh-en/IWSLT17.TED.tst2017.zh-en.zh.xml",
            ],
            "zh-en": [
                "zh-en/IWSLT17.TED.tst2017.zh-en.zh.xml",
                "en-zh/IWSLT17.TED.tst2017.en-zh.en.xml",
            ],
            "en-ar": [
                "en-ar/IWSLT17.TED.tst2017.en-ar.en.xml",
                "ar-en/IWSLT17.TED.tst2017.ar-en.ar.xml",
            ],
            "ar-en": [
                "ar-en/IWSLT17.TED.tst2017.ar-en.ar.xml",
                "en-ar/IWSLT17.TED.tst2017.en-ar.en.xml",
            ],
            "en-ja": [
                "en-ja/IWSLT17.TED.tst2017.en-ja.en.xml",
                "ja-en/IWSLT17.TED.tst2017.ja-en.ja.xml",
            ],
            "ja-en": [
                "ja-en/IWSLT17.TED.tst2017.ja-en.ja.xml",
                "en-ja/IWSLT17.TED.tst2017.en-ja.en.xml",
            ],
            "en-ko": [
                "en-ko/IWSLT17.TED.tst2017.en-ko.en.xml",
                "ko-en/IWSLT17.TED.tst2017.ko-en.ko.xml",
            ],
            "ko-en": [
                "ko-en/IWSLT17.TED.tst2017.ko-en.ko.xml",
                "en-ko/IWSLT17.TED.tst2017.en-ko.en.xml",
            ],
        },
    ),
    "iwslt17/tst2016": IWSLTXMLDataset(
        "iwslt17/tst2016",
        data=[
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/en/fr/en-fr.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/fr/en/fr-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/en/de/en-de.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/de/en/de-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/en/zh/en-zh.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-ted-test/texts/zh/en/zh-en.tgz",
        ],
        md5=[
            "1849bcc3b006dc0642a8843b11aa7192",
            "79bf7a2ef02d226875f55fb076e7e473",
            "b68e7097b179491f6c466ef41ad72b9b",
            "e3f5b2a075a2da1a395c8b60bf1e9be1",
            "975a858783a0ebec8c57d83ddd5bd381",
            "cc51d9b7fe1ff2af858c6a0dd80b8815",
        ],
        description="Development data for IWSLT 2017.",
        langpairs={
            "en-fr": [
                "en-fr/IWSLT17.TED.tst2016.en-fr.en.xml",
                "fr-en/IWSLT17.TED.tst2016.fr-en.fr.xml",
            ],
            "fr-en": [
                "fr-en/IWSLT17.TED.tst2016.fr-en.fr.xml",
                "en-fr/IWSLT17.TED.tst2016.en-fr.en.xml",
            ],
            "en-de": [
                "en-de/IWSLT17.TED.tst2016.en-de.en.xml",
                "de-en/IWSLT17.TED.tst2016.de-en.de.xml",
            ],
            "de-en": [
                "de-en/IWSLT17.TED.tst2016.de-en.de.xml",
                "en-de/IWSLT17.TED.tst2016.en-de.en.xml",
            ],
            "en-zh": [
                "en-zh/IWSLT17.TED.tst2016.en-zh.en.xml",
                "zh-en/IWSLT17.TED.tst2016.zh-en.zh.xml",
            ],
            "zh-en": [
                "zh-en/IWSLT17.TED.tst2016.zh-en.zh.xml",
                "en-zh/IWSLT17.TED.tst2016.en-zh.en.xml",
            ],
        },
    ),
    "iwslt17/tst2015": IWSLTXMLDataset(
        "iwslt17/tst2015",
        data=[
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/de/en-de.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/de/en/de-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/fr/en-fr.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/fr/en/fr-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/zh/en-zh.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/zh/en/zh-en.tgz",
        ],
        md5=[
            "d8a32cfc002a4f12b17429cfa78050e6",
            "ca2b94d694150d4d6c5dc64c200fa589",
            "3cf07ebe305312b12f7f1a4d5f8f8377",
            "19927da9de0f40348cad9c0fc61642ac",
            "575b788dad6c5b9c5cee636f9ac1094a",
            "1c0ae40171d52593df8a6963d3828116",
        ],
        description="Development data for IWSLT 2017.",
        langpairs={
            "en-fr": [
                "en-fr/IWSLT17.TED.tst2015.en-fr.en.xml",
                "fr-en/IWSLT17.TED.tst2015.fr-en.fr.xml",
            ],
            "fr-en": [
                "fr-en/IWSLT17.TED.tst2015.fr-en.fr.xml",
                "en-fr/IWSLT17.TED.tst2015.en-fr.en.xml",
            ],
            "en-de": [
                "en-de/IWSLT17.TED.tst2015.en-de.en.xml",
                "de-en/IWSLT17.TED.tst2015.de-en.de.xml",
            ],
            "de-en": [
                "de-en/IWSLT17.TED.tst2015.de-en.de.xml",
                "en-de/IWSLT17.TED.tst2015.en-de.en.xml",
            ],
            "en-zh": [
                "en-zh/IWSLT17.TED.tst2015.en-zh.en.xml",
                "zh-en/IWSLT17.TED.tst2015.zh-en.zh.xml",
            ],
            "zh-en": [
                "zh-en/IWSLT17.TED.tst2015.zh-en.zh.xml",
                "en-zh/IWSLT17.TED.tst2015.en-zh.en.xml",
            ],
        },
    ),
    "iwslt17/tst2014": IWSLTXMLDataset(
        "iwslt17/tst2014",
        data=[
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/de/en-de.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/de/en/de-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/fr/en-fr.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/fr/en/fr-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/zh/en-zh.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/zh/en/zh-en.tgz",
        ],
        md5=[
            "d8a32cfc002a4f12b17429cfa78050e6",
            "ca2b94d694150d4d6c5dc64c200fa589",
            "3cf07ebe305312b12f7f1a4d5f8f8377",
            "19927da9de0f40348cad9c0fc61642ac",
            "575b788dad6c5b9c5cee636f9ac1094a",
            "1c0ae40171d52593df8a6963d3828116",
        ],
        description="Development data for IWSLT 2017.",
        langpairs={
            "en-fr": [
                "en-fr/IWSLT17.TED.tst2014.en-fr.en.xml",
                "fr-en/IWSLT17.TED.tst2014.fr-en.fr.xml",
            ],
            "fr-en": [
                "fr-en/IWSLT17.TED.tst2014.fr-en.fr.xml",
                "en-fr/IWSLT17.TED.tst2014.en-fr.en.xml",
            ],
            "en-de": [
                "en-de/IWSLT17.TED.tst2014.en-de.en.xml",
                "de-en/IWSLT17.TED.tst2014.de-en.de.xml",
            ],
            "de-en": [
                "de-en/IWSLT17.TED.tst2014.de-en.de.xml",
                "en-de/IWSLT17.TED.tst2014.en-de.en.xml",
            ],
            "en-zh": [
                "en-zh/IWSLT17.TED.tst2014.en-zh.en.xml",
                "zh-en/IWSLT17.TED.tst2014.zh-en.zh.xml",
            ],
            "zh-en": [
                "zh-en/IWSLT17.TED.tst2014.zh-en.zh.xml",
                "en-zh/IWSLT17.TED.tst2014.en-zh.en.xml",
            ],
        },
    ),
    "iwslt17/tst2013": IWSLTXMLDataset(
        "iwslt17/tst2013",
        data=[
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/de/en-de.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/de/en/de-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/fr/en-fr.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/fr/en/fr-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/zh/en-zh.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/zh/en/zh-en.tgz",
        ],
        md5=[
            "d8a32cfc002a4f12b17429cfa78050e6",
            "ca2b94d694150d4d6c5dc64c200fa589",
            "3cf07ebe305312b12f7f1a4d5f8f8377",
            "19927da9de0f40348cad9c0fc61642ac",
            "575b788dad6c5b9c5cee636f9ac1094a",
            "1c0ae40171d52593df8a6963d3828116",
        ],
        description="Development data for IWSLT 2017.",
        langpairs={
            "en-fr": [
                "en-fr/IWSLT17.TED.tst2013.en-fr.en.xml",
                "fr-en/IWSLT17.TED.tst2013.fr-en.fr.xml",
            ],
            "fr-en": [
                "fr-en/IWSLT17.TED.tst2013.fr-en.fr.xml",
                "en-fr/IWSLT17.TED.tst2013.en-fr.en.xml",
            ],
            "en-de": [
                "en-de/IWSLT17.TED.tst2013.en-de.en.xml",
                "de-en/IWSLT17.TED.tst2013.de-en.de.xml",
            ],
            "de-en": [
                "de-en/IWSLT17.TED.tst2013.de-en.de.xml",
                "en-de/IWSLT17.TED.tst2013.en-de.en.xml",
            ],
            "en-zh": [
                "en-zh/IWSLT17.TED.tst2013.en-zh.en.xml",
                "zh-en/IWSLT17.TED.tst2013.zh-en.zh.xml",
            ],
            "zh-en": [
                "zh-en/IWSLT17.TED.tst2013.zh-en.zh.xml",
                "en-zh/IWSLT17.TED.tst2013.en-zh.en.xml",
            ],
        },
    ),
    "iwslt17/tst2012": IWSLTXMLDataset(
        "iwslt17/tst2012",
        data=[
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/de/en-de.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/de/en/de-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/fr/en-fr.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/fr/en/fr-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/zh/en-zh.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/zh/en/zh-en.tgz",
        ],
        md5=[
            "d8a32cfc002a4f12b17429cfa78050e6",
            "ca2b94d694150d4d6c5dc64c200fa589",
            "3cf07ebe305312b12f7f1a4d5f8f8377",
            "19927da9de0f40348cad9c0fc61642ac",
            "575b788dad6c5b9c5cee636f9ac1094a",
            "1c0ae40171d52593df8a6963d3828116",
        ],
        description="Development data for IWSLT 2017.",
        langpairs={
            "en-fr": [
                "en-fr/IWSLT17.TED.tst2012.en-fr.en.xml",
                "fr-en/IWSLT17.TED.tst2012.fr-en.fr.xml",
            ],
            "fr-en": [
                "fr-en/IWSLT17.TED.tst2012.fr-en.fr.xml",
                "en-fr/IWSLT17.TED.tst2012.en-fr.en.xml",
            ],
            "en-de": [
                "en-de/IWSLT17.TED.tst2012.en-de.en.xml",
                "de-en/IWSLT17.TED.tst2012.de-en.de.xml",
            ],
            "de-en": [
                "de-en/IWSLT17.TED.tst2012.de-en.de.xml",
                "en-de/IWSLT17.TED.tst2012.en-de.en.xml",
            ],
            "en-zh": [
                "en-zh/IWSLT17.TED.tst2012.en-zh.en.xml",
                "zh-en/IWSLT17.TED.tst2012.zh-en.zh.xml",
            ],
            "zh-en": [
                "zh-en/IWSLT17.TED.tst2012.zh-en.zh.xml",
                "en-zh/IWSLT17.TED.tst2012.en-zh.en.xml",
            ],
        },
    ),
    "iwslt17/tst2011": IWSLTXMLDataset(
        "iwslt17/tst2011",
        data=[
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/de/en-de.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/de/en/de-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/fr/en-fr.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/fr/en/fr-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/zh/en-zh.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/zh/en/zh-en.tgz",
        ],
        md5=[
            "d8a32cfc002a4f12b17429cfa78050e6",
            "ca2b94d694150d4d6c5dc64c200fa589",
            "3cf07ebe305312b12f7f1a4d5f8f8377",
            "19927da9de0f40348cad9c0fc61642ac",
            "575b788dad6c5b9c5cee636f9ac1094a",
            "1c0ae40171d52593df8a6963d3828116",
        ],
        description="Development data for IWSLT 2017.",
        langpairs={
            "en-fr": [
                "en-fr/IWSLT17.TED.tst2011.en-fr.en.xml",
                "fr-en/IWSLT17.TED.tst2011.fr-en.fr.xml",
            ],
            "fr-en": [
                "fr-en/IWSLT17.TED.tst2011.fr-en.fr.xml",
                "en-fr/IWSLT17.TED.tst2011.en-fr.en.xml",
            ],
            "en-de": [
                "en-de/IWSLT17.TED.tst2011.en-de.en.xml",
                "de-en/IWSLT17.TED.tst2011.de-en.de.xml",
            ],
            "de-en": [
                "de-en/IWSLT17.TED.tst2011.de-en.de.xml",
                "en-de/IWSLT17.TED.tst2011.en-de.en.xml",
            ],
            "en-zh": [
                "en-zh/IWSLT17.TED.tst2011.en-zh.en.xml",
                "zh-en/IWSLT17.TED.tst2011.zh-en.zh.xml",
            ],
            "zh-en": [
                "zh-en/IWSLT17.TED.tst2011.zh-en.zh.xml",
                "en-zh/IWSLT17.TED.tst2011.en-zh.en.xml",
            ],
        },
    ),
    "iwslt17/tst2010": IWSLTXMLDataset(
        "iwslt17/tst2010",
        data=[
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/de/en-de.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/de/en/de-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/fr/en-fr.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/fr/en/fr-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/zh/en-zh.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/zh/en/zh-en.tgz",
        ],
        md5=[
            "d8a32cfc002a4f12b17429cfa78050e6",
            "ca2b94d694150d4d6c5dc64c200fa589",
            "3cf07ebe305312b12f7f1a4d5f8f8377",
            "19927da9de0f40348cad9c0fc61642ac",
            "575b788dad6c5b9c5cee636f9ac1094a",
            "1c0ae40171d52593df8a6963d3828116",
        ],
        description="Development data for IWSLT 2017.",
        langpairs={
            "en-fr": [
                "en-fr/IWSLT17.TED.tst2010.en-fr.en.xml",
                "fr-en/IWSLT17.TED.tst2010.fr-en.fr.xml",
            ],
            "fr-en": [
                "fr-en/IWSLT17.TED.tst2010.fr-en.fr.xml",
                "en-fr/IWSLT17.TED.tst2010.en-fr.en.xml",
            ],
            "en-de": [
                "en-de/IWSLT17.TED.tst2010.en-de.en.xml",
                "de-en/IWSLT17.TED.tst2010.de-en.de.xml",
            ],
            "de-en": [
                "de-en/IWSLT17.TED.tst2010.de-en.de.xml",
                "en-de/IWSLT17.TED.tst2010.en-de.en.xml",
            ],
            "en-zh": [
                "en-zh/IWSLT17.TED.tst2010.en-zh.en.xml",
                "zh-en/IWSLT17.TED.tst2010.zh-en.zh.xml",
            ],
            "zh-en": [
                "zh-en/IWSLT17.TED.tst2010.zh-en.zh.xml",
                "en-zh/IWSLT17.TED.tst2010.en-zh.en.xml",
            ],
        },
    ),
    "iwslt17/dev2010": IWSLTXMLDataset(
        "iwslt17/dev2010",
        data=[
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/de/en-de.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/de/en/de-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/fr/en-fr.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/fr/en/fr-en.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/en/zh/en-zh.tgz",
            "https://raw.githubusercontent.com/hlt-mt/WIT3/master/archive/2017-01-trnted/texts/zh/en/zh-en.tgz",
        ],
        md5=[
            "d8a32cfc002a4f12b17429cfa78050e6",
            "ca2b94d694150d4d6c5dc64c200fa589",
            "3cf07ebe305312b12f7f1a4d5f8f8377",
            "19927da9de0f40348cad9c0fc61642ac",
            "575b788dad6c5b9c5cee636f9ac1094a",
            "1c0ae40171d52593df8a6963d3828116",
        ],
        description="Development data for IWSLT 2017.",
        langpairs={
            "en-fr": [
                "en-fr/IWSLT17.TED.dev2010.en-fr.en.xml",
                "fr-en/IWSLT17.TED.dev2010.fr-en.fr.xml",
            ],
            "fr-en": [
                "fr-en/IWSLT17.TED.dev2010.fr-en.fr.xml",
                "en-fr/IWSLT17.TED.dev2010.en-fr.en.xml",
            ],
            "en-de": [
                "en-de/IWSLT17.TED.dev2010.en-de.en.xml",
                "de-en/IWSLT17.TED.dev2010.de-en.de.xml",
            ],
            "de-en": [
                "de-en/IWSLT17.TED.dev2010.de-en.de.xml",
                "en-de/IWSLT17.TED.dev2010.en-de.en.xml",
            ],
            "en-zh": [
                "en-zh/IWSLT17.TED.dev2010.en-zh.en.xml",
                "zh-en/IWSLT17.TED.dev2010.zh-en.zh.xml",
            ],
            "zh-en": [
                "zh-en/IWSLT17.TED.dev2010.zh-en.zh.xml",
                "en-zh/IWSLT17.TED.dev2010.en-zh.en.xml",
            ],
        },
    ),
    # mtedx
    "mtedx/valid": PlainTextDataset(
        "mtedx/valid",
        data=[
            "https://raw.githubusercontent.com/esalesky/mtedx-eval/main/valid.tar.gz"
        ],
        description="mTEDx evaluation data, valid: http://openslr.org/100",
        citation="@misc{salesky2021multilingual,\n      title={The Multilingual TEDx Corpus for Speech Recognition and Translation}, \n      author={Elizabeth Salesky and Matthew Wiesner and Jacob Bremerman and Roldano Cattoni and Matteo Negri and Marco Turchi and Douglas W. Oard and Matt Post},\n      year={2021},\n      eprint={2102.01757},\n      archivePrefix={arXiv},\n      primaryClass={cs.CL}\n}",
        md5=["40618171614c50e6cbb5e5bbceee0635"],
        langpairs={
            "el-en": ["valid/mtedx-valid-elen.el", "valid/mtedx-valid-elen.en"],
            "es-en": ["valid/mtedx-valid-esen.es", "valid/mtedx-valid-esen.en"],
            "es-fr": ["valid/mtedx-valid-esfr.es", "valid/mtedx-valid-esfr.fr"],
            "es-it": ["valid/mtedx-valid-esit.es", "valid/mtedx-valid-esit.it"],
            "es-pt": ["valid/mtedx-valid-espt.es", "valid/mtedx-valid-espt.pt"],
            "fr-en": ["valid/mtedx-valid-fren.fr", "valid/mtedx-valid-fren.en"],
            "fr-es": ["valid/mtedx-valid-fres.fr", "valid/mtedx-valid-fres.es"],
            "fr-pt": ["valid/mtedx-valid-frpt.fr", "valid/mtedx-valid-frpt.pt"],
            "it-en": ["valid/mtedx-valid-iten.it", "valid/mtedx-valid-iten.en"],
            "it-es": ["valid/mtedx-valid-ites.it", "valid/mtedx-valid-ites.es"],
            "pt-en": ["valid/mtedx-valid-pten.pt", "valid/mtedx-valid-pten.en"],
            "pt-es": ["valid/mtedx-valid-ptes.pt", "valid/mtedx-valid-ptes.es"],
            "ru-en": ["valid/mtedx-valid-ruen.ru", "valid/mtedx-valid-ruen.en"],
        },
    ),
    "mtedx/test": PlainTextDataset(
        "mtedx/test",
        data=["https://raw.githubusercontent.com/esalesky/mtedx-eval/main/test.tar.gz"],
        description="mTEDx evaluation data, test: http://openslr.org/100",
        citation="@misc{salesky2021multilingual,\n      title={The Multilingual TEDx Corpus for Speech Recognition and Translation}, \n      author={Elizabeth Salesky and Matthew Wiesner and Jacob Bremerman and Roldano Cattoni and Matteo Negri and Marco Turchi and Douglas W. Oard and Matt Post},\n      year={2021},\n      eprint={2102.01757},\n      archivePrefix={arXiv},\n      primaryClass={cs.CL}\n}",
        md5=["fa4cb1548c210ec424d7d6bc9a3675a7"],
        langpairs={
            "el-en": ["test/mtedx-test-elen.el", "test/mtedx-test-elen.en"],
            "es-en": ["test/mtedx-test-esen.es", "test/mtedx-test-esen.en"],
            "es-fr": ["test/mtedx-test-esfr.es", "test/mtedx-test-esfr.fr"],
            "es-it": ["test/mtedx-test-esit.es", "test/mtedx-test-esit.it"],
            "es-pt": ["test/mtedx-test-espt.es", "test/mtedx-test-espt.pt"],
            "fr-en": ["test/mtedx-test-fren.fr", "test/mtedx-test-fren.en"],
            "fr-es": ["test/mtedx-test-fres.fr", "test/mtedx-test-fres.es"],
            "fr-pt": ["test/mtedx-test-frpt.fr", "test/mtedx-test-frpt.pt"],
            "it-en": ["test/mtedx-test-iten.it", "test/mtedx-test-iten.en"],
            "it-es": ["test/mtedx-test-ites.it", "test/mtedx-test-ites.es"],
            "pt-en": ["test/mtedx-test-pten.pt", "test/mtedx-test-pten.en"],
            "pt-es": ["test/mtedx-test-ptes.pt", "test/mtedx-test-ptes.es"],
            "ru-en": ["test/mtedx-test-ruen.ru", "test/mtedx-test-ruen.en"],
        },
    ),
    # multi30k
    "multi30k/2016": PlainTextDataset(
        "multi30k/2016",
        data=[
            "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/multi30k_test_sets_d3ec2a38.tar.gz"
        ],
        md5=["9cf8f22d57fee2ca2af3c682dfdc525b"],
        description="2016 flickr test set of Multi30k dataset",
        citation='@InProceedings{elliott-etal-2016-multi30k,\n    title = "{M}ulti30{K}: Multilingual {E}nglish-{G}erman Image Descriptions",\n    author = "Elliott, Desmond  and Frank, Stella  and Sima{\'}an, Khalil  and Specia, Lucia",\n    booktitle = "Proceedings of the 5th Workshop on Vision and Language",\n    month = aug,\n    year = "2016",\n    address = "Berlin, Germany",\n    publisher = "Association for Computational Linguistics",\n    url = "https://www.aclweb.org/anthology/W16-3210",\n    doi = "10.18653/v1/W16-3210",\n    pages = "70--74",\n}',
        langpairs={
            "en-fr": ["test_2016_flickr.en", "test_2016_flickr.fr"],
            "en-de": ["test_2016_flickr.en", "test_2016_flickr.de"],
            "en-cs": ["test_2016_flickr.en", "test_2016_flickr.cs"],
        },
    ),
    "multi30k/2017": PlainTextDataset(
        "multi30k/2017",
        data=[
            "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/multi30k_test_sets_d3ec2a38.tar.gz"
        ],
        md5=["9cf8f22d57fee2ca2af3c682dfdc525b"],
        description="2017 flickr test set of Multi30k dataset",
        citation='@InProceedings{elliott-etal-2016-multi30k,\n    title = "{M}ulti30{K}: Multilingual {E}nglish-{G}erman Image Descriptions",\n    author = "Elliott, Desmond  and Frank, Stella  and Sima{\'}an, Khalil  and Specia, Lucia",\n    booktitle = "Proceedings of the 5th Workshop on Vision and Language",\n    month = aug,\n    year = "2016",\n    address = "Berlin, Germany",\n    publisher = "Association for Computational Linguistics",\n    url = "https://www.aclweb.org/anthology/W16-3210",\n    doi = "10.18653/v1/W16-3210",\n    pages = "70--74",\n}\n\n@InProceedings{elliott-etal-2017-findings,\n    title = "Findings of the Second Shared Task on Multimodal Machine Translation and Multilingual Image Description",\n    author = {Elliott, Desmond  and Frank, Stella  and Barrault, Lo{\\"\\i}c  and Bougares, Fethi  and Specia, Lucia},\n    booktitle = "Proceedings of the Second Conference on Machine Translation",\n    month = sep,\n    year = "2017",\n    address = "Copenhagen, Denmark",\n    publisher = "Association for Computational Linguistics",\n    url = "https://www.aclweb.org/anthology/W17-4718",\n    doi = "10.18653/v1/W17-4718",\n    pages = "215--233",\n}\n',
        langpairs={
            "en-fr": ["test_2017_flickr.en", "test_2017_flickr.fr"],
            "en-de": ["test_2017_flickr.en", "test_2017_flickr.de"],
        },
    ),
    "multi30k/2018": PlainTextDataset(
        "multi30k/2018",
        data=[
            "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/multi30k_test_sets_d3ec2a38.tar.gz",
            "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2018_flickr.cs.gz",
            "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2018_flickr.de.gz",
            "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2018_flickr.fr.gz",
        ],
        md5=[
            "9cf8f22d57fee2ca2af3c682dfdc525b",
            "4c6b6490e58107b2e397c5e3e1690abc",
            "87e00327083dd69feaa029a8f7c1a047",
            "a64563e986438ed731a6713027c36bfd",
        ],
        description="2018 flickr test set of Multi30k dataset. See https://competitions.codalab.org/competitions/19917 for evaluation.",
        citation='@InProceedings{elliott-etal-2016-multi30k,\n    title = "{M}ulti30{K}: Multilingual {E}nglish-{G}erman Image Descriptions",\n    author = "Elliott, Desmond  and Frank, Stella  and Sima{\'}an, Khalil  and Specia, Lucia",\n    booktitle = "Proceedings of the 5th Workshop on Vision and Language",\n    month = aug,\n    year = "2016",\n    address = "Berlin, Germany",\n    publisher = "Association for Computational Linguistics",\n    url = "https://www.aclweb.org/anthology/W16-3210",\n    doi = "10.18653/v1/W16-3210",\n    pages = "70--74",\n}\n\n@InProceedings{barrault-etal-2018-findings,\n    title = "Findings of the Third Shared Task on Multimodal Machine Translation",\n    author = {Barrault, Lo{\\"\\i}c  and Bougares, Fethi  and Specia, Lucia  and Lala, Chiraag  and Elliott, Desmond  and Frank, Stella},\n    booktitle = "Proceedings of the Third Conference on Machine Translation: Shared Task Papers",\n    month = oct,\n    year = "2018",\n    address = "Belgium, Brussels",\n    publisher = "Association for Computational Linguistics",\n    url = "https://www.aclweb.org/anthology/W18-6402",\n    doi = "10.18653/v1/W18-6402",\n    pages = "304--323",\n}\n',
        langpairs={
            "en-fr": ["test_2018_flickr.en", "multi30k_2018.test_2018_flickr.fr.gz"],
            "en-de": ["test_2018_flickr.en", "multi30k_2018.test_2018_flickr.de.gz"],
            "en-cs": ["test_2018_flickr.en", "multi30k_2018.test_2018_flickr.cs.gz"],
        },
    ),
    # mtnt
    "mtnt2019": TSVDataset(
        "mtnt2019",
        data=["https://pmichel31415.github.io/hosting/MTNT2019.tar.gz"],
        description="Test set for the WMT 19 robustness shared task",
        md5=["78a672e1931f106a8549023c0e8af8f6"],
        langpairs={
            "en-fr": ["2:MTNT2019/en-fr.final.tsv", "3:MTNT2019/en-fr.final.tsv"],
            "fr-en": ["2:MTNT2019/fr-en.final.tsv", "3:MTNT2019/fr-en.final.tsv"],
            "en-ja": ["2:MTNT2019/en-ja.final.tsv", "3:MTNT2019/en-ja.final.tsv"],
            "ja-en": ["2:MTNT2019/ja-en.final.tsv", "3:MTNT2019/ja-en.final.tsv"],
        },
    ),
    "mtnt1.1/test": TSVDataset(
        "mtnt1.1/test",
        data=[
            "https://github.com/pmichel31415/mtnt/releases/download/v1.1/MTNT.1.1.tar.gz"
        ],
        description="Test data for the Machine Translation of Noisy Text task: http://www.cs.cmu.edu/~pmichel1/mtnt/",
        citation='@InProceedings{michel2018a:mtnt,\n    author = "Michel, Paul and Neubig, Graham",\n    title = "MTNT: A Testbed for Machine Translation of Noisy Text",\n    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",\n    year = "2018",\n    publisher = "Association for Computational Linguistics",\n    pages = "543--553",\n    location = "Brussels, Belgium",\n    url = "http://aclweb.org/anthology/D18-1050"\n}',
        md5=["8ce1831ac584979ba8cdcd9d4be43e1d"],
        langpairs={
            "en-fr": ["1:MTNT/test/test.en-fr.tsv", "2:MTNT/test/test.en-fr.tsv"],
            "fr-en": ["1:MTNT/test/test.fr-en.tsv", "2:MTNT/test/test.fr-en.tsv"],
            "en-ja": ["1:MTNT/test/test.en-ja.tsv", "2:MTNT/test/test.en-ja.tsv"],
            "ja-en": ["1:MTNT/test/test.ja-en.tsv", "2:MTNT/test/test.ja-en.tsv"],
        },
    ),
    "mtnt1.1/valid": TSVDataset(
        "mtnt1.1/valid",
        data=[
            "https://github.com/pmichel31415/mtnt/releases/download/v1.1/MTNT.1.1.tar.gz"
        ],
        description="Validation data for the Machine Translation of Noisy Text task: http://www.cs.cmu.edu/~pmichel1/mtnt/",
        citation='@InProceedings{michel2018a:mtnt,\n    author = "Michel, Paul and Neubig, Graham",\n    title = "MTNT: A Testbed for Machine Translation of Noisy Text",\n    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",\n    year = "2018",\n    publisher = "Association for Computational Linguistics",\n    pages = "543--553",\n    location = "Brussels, Belgium",\n    url = "http://aclweb.org/anthology/D18-1050"\n}',
        md5=["8ce1831ac584979ba8cdcd9d4be43e1d"],
        langpairs={
            "en-fr": ["1:MTNT/valid/valid.en-fr.tsv", "2:MTNT/valid/valid.en-fr.tsv"],
            "fr-en": ["1:MTNT/valid/valid.fr-en.tsv", "2:MTNT/valid/valid.fr-en.tsv"],
            "en-ja": ["1:MTNT/valid/valid.en-ja.tsv", "2:MTNT/valid/valid.en-ja.tsv"],
            "ja-en": ["1:MTNT/valid/valid.ja-en.tsv", "2:MTNT/valid/valid.ja-en.tsv"],
        },
    ),
    "mtnt1.1/train": TSVDataset(
        "mtnt1.1/train",
        data=[
            "https://github.com/pmichel31415/mtnt/releases/download/v1.1/MTNT.1.1.tar.gz"
        ],
        description="Validation data for the Machine Translation of Noisy Text task: http://www.cs.cmu.edu/~pmichel1/mtnt/",
        citation='@InProceedings{michel2018a:mtnt,\n    author = "Michel, Paul and Neubig, Graham",\n    title = "MTNT: A Testbed for Machine Translation of Noisy Text",\n    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",\n    year = "2018",\n    publisher = "Association for Computational Linguistics",\n    pages = "543--553",\n    location = "Brussels, Belgium",\n    url = "http://aclweb.org/anthology/D18-1050"\n}',
        md5=["8ce1831ac584979ba8cdcd9d4be43e1d"],
        langpairs={
            "en-fr": ["1:MTNT/train/train.en-fr.tsv", "2:MTNT/train/train.en-fr.tsv"],
            "fr-en": ["1:MTNT/train/train.fr-en.tsv", "2:MTNT/train/train.fr-en.tsv"],
            "en-ja": ["1:MTNT/train/train.en-ja.tsv", "2:MTNT/train/train.en-ja.tsv"],
            "ja-en": ["1:MTNT/train/train.ja-en.tsv", "2:MTNT/train/train.ja-en.tsv"],
        },
    ),
}
