#!/bin/bash

if [[ -z $1 ]]; then
    echo "Usage: add_wmt.sh TARBALL_URL"
    cat<<EOF
Downloads tarball and generates the Python code used to add a new dataset.
EOF
fi

tarball=$1

name=wmt$(date +%y)

wget -q -O $name.tgz $tarball
md5=$(md5sum $name.tgz | cut -d' ' -f1)
tar xzvf $name.tgz

langs=$(ls sgm/ | cut -d- -f2 | sort -u)

echo "    '$name': {"
echo "        'data': ['$1'],"
echo "        'md5': ['$md5'],"
for entry in $langs; do
    # pair has form "zhen", turn to "zh-en"
    source=${entry:0:2}
    target=${entry:2:2}
    pair=$source-$target
    source=$(ls sgm/*$entry-src.$source.sgm)
    target=$(ls sgm/*$entry-ref.$target.sgm)
    echo "        '$pair': ['$source', '$target'],"
done
echo "    },"
