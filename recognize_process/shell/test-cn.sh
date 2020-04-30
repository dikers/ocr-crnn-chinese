DATA_TARGET_DIR='./data_cn/'

# 测试的类型  train valid
TEST_TYPE='valid'

export PYTHONPATH=../../
python3 ../tools/test_crnn.py \
-i=${DATA_TARGET_DIR}'images/'${TEST_TYPE} \
-w=${DATA_TARGET_DIR}'model_save' \
-c=${DATA_TARGET_DIR}'char_map.json' \
-t=${DATA_TARGET_DIR}${TEST_TYPE}'_labels.txt'