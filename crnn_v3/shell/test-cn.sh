DATA_TARGET_DIR='./data_cn/'
export PYTHONPATH=../

# 测试的类型  train valid
TEST_TYPE='train'

python3 ../tools/test_shadownet.py \
--image_path ${DATA_TARGET_DIR}'images/'${TEST_TYPE}'/' \
--weights_path ${DATA_TARGET_DIR}'model_save' \
--char_dict_path ${DATA_TARGET_DIR}'char_dict.json' \
--ord_map_dict_path ${DATA_TARGET_DIR}'ord_map.json' \
--txt_path ${DATA_TARGET_DIR}${TEST_TYPE}'_labels.txt' \
--visualize 0