#!/usr/bin/env bash
export PYTHONPATH=./
python3 './train_model/tools/train_crnn.py' \
-train='output/tfrecords/train' \
-val='output/tfrecords/valid' \
-c='output/text_data/char_map.json' \
-s='output/model_save'