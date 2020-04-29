DATA_TARGET_DIR='./data_cn/'
export PYTHONPATH=../../
python3 ../tools/test_crnn.py \
-i=${DATA_TARGET_DIR}'images/train' \
-w=${DATA_TARGET_DIR}'model_save' \
-c=${DATA_TARGET_DIR}'char_map.json' \
-t=${DATA_TARGET_DIR}'test_labels.txt'