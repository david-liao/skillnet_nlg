#!/usr/bin/env bash

predict_metrics_file=$1

# nlpcc
pushd /usr/bin
ln -snf python2 python
popd

pushd scripts/eval/nlpcc2018_postprocess/
python3 convert.py ../../../$predict_metrics_file
popd

pushd /usr/bin
ln -snf python3 python
popd