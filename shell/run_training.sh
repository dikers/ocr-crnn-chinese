#!/usr/bin/env bash
export PYTHONPATH=/Users/liujunyi/Documents/git/ocr-crnn-tensorflow
python3 source/step_three_recognize_process/tools/train_crnn.py \
-d='source/temp/test_write_tfrecord/tfrecord' \
-c='source/temp/test_write_tfrecord/temp_data/char_map_new.json' \
-s='source/temp/test_write_tfrecord/model_save'