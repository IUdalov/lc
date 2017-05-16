#!/usr/bin/env bash
set -e

export LC_STEPS=10

for dataset in $@; do
    echo "--------------------------------------------------------------------------------"
    echo "Data set $dataset"
    #time bin/svm-train -t 0 $dataset /tmp/sss &> 1
    time bin/lc-train $dataset /tmp/sss
    echo "--------------------------------------------------------------------------------"
done