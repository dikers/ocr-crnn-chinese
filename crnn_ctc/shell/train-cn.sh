DATA_TARGET_DIR='./data_cn/'
export PYTHONPATH=../

python3 ../tools/train_shadownet.py \
--dataset_dir ${DATA_TARGET_DIR} \
--char_dict_path ${DATA_TARGET_DIR}'char_dict.json' \
--ord_map_dict_path ${DATA_TARGET_DIR}'ord_map.json' \
--weights_path ${DATA_TARGET_DIR}'model_save' \
--model_save_dir ${DATA_TARGET_DIR}'model_save' \
--multi_gpus 1