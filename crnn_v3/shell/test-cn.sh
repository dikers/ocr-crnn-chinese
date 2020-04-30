DATA_TARGET_DIR='./data_cn/'
export PYTHONPATH=../

python3 ../tools/test_shadownet.py \
--image_path ${DATA_TARGET_DIR}'images/train/' \
--weights_path ${DATA_TARGET_DIR}'model_save' \
--char_dict_path ${DATA_TARGET_DIR}'char_dict.json' \
--ord_map_dict_path ${DATA_TARGET_DIR}'ord_map.json' \
--txt_path ${DATA_TARGET_DIR}'train_labels.txt' \
--visualize 0