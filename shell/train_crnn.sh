#!/usr/bin/env bash
export PYTHONPATH=./
python3 './train_model/tools/train_crnn.py' \
-i='output/images/train' \
-c='output/text_data/char_map.json' \
-s='output/model_save' \
-w='output/model_save'