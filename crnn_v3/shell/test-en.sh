DATA_TARGET_DIR='../../train_model/shell/data_cn/'

export PYTHONPATH=../


python ../tools/test_shadownet.py \
--dataset_dir ${DATA_TARGET_DIR}'label_test.txt' \
--weights_path ${DATA_TARGET_DIR}'model_save' \
--char_dict_path ${DATA_TARGET_DIR}'char_dict.json' \
--ord_map_dict_path ${DATA_TARGET_DIR}'ord_map.json' 