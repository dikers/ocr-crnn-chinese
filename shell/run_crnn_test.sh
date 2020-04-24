#!/usr/bin/env bash
export PYTHONPATH=/Users/liujunyi/Documents/git/ocr-crnn-tensorflow
##--weights_path='source/core/step_three_recognize_process/model_save/recognize_model'
python source/core/step_three_recognize_process/tools/test_ctc_debug.py \
--image_path='source/core/step_three_recognize_process/test_imgs/17_1972_4.jpg' \
--weights_path='source/core/step_three_recognize_process/model_synth90_121999/shadownet_2020-03-25-04-04-17.ckpt-121999'