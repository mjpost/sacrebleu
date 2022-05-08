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


# This defines data locations.
# At the top level are test sets.
# Beneath each test set, we define the location to download the test data.
# The other keys are each language pair contained in the tarball, and the respective locations of the source and reference data within each.
# Many of these are *.sgm files, which are processed to produced plain text that can be used by this script.
# The canonical location of unpacked, processed data is $SACREBLEU_DIR/$TEST/$SOURCE-$TARGET.{$SOURCE,$TARGET}
DATASETS = {}

from .fake_sgml import FAKE_SGML_DATASETS
from .iwslt_xml import IWSLT_XML_DATASETS
from .plain_text import PLAIN_TEXT_DATASETS
from .tsv import TSV_DATASETS
from .wmt_xml import WMT_XML_DATASETS

for item in [
    FAKE_SGML_DATASETS,
    PLAIN_TEXT_DATASETS,
    WMT_XML_DATASETS,
    IWSLT_XML_DATASETS,
    TSV_DATASETS,
]:
    DATASETS.update(item)
