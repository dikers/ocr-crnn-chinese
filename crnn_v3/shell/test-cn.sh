#DATA_TARGET_DIR='../../train_model/shell/data_cn/'
DATA_TARGET_DIR='/home/ec2-user/tfc/ocr_project/ocr-crnn-chinese/train_model/shell/data_cn/'
export PYTHONPATH=../

python3 ../tools/test_shadownet.py \
--image_path ${DATA_TARGET_DIR}'images/valid/' \
--weights_path ${DATA_TARGET_DIR}'model_save' \
--char_dict_path ${DATA_TARGET_DIR}'char_dict.json' \
--ord_map_dict_path ${DATA_TARGET_DIR}'ord_map.json' \
--txt_path ${DATA_TARGET_DIR}'valid_labels.txt' \
--visualize 0