DATA_TARGET_DIR='./data_en/'
# 下载地址 http://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz
DATA_SRC_DIR='../../output/mjsynth_data/mnt/ramdisk/max/90kDICT32px/'
export PYTHONPATH=../

python3 ../tools/test_shadownet.py \
--image_path ${DATA_SRC_DIR} \
--weights_path ${DATA_TARGET_DIR}'model_save' \
--char_dict_path ${DATA_TARGET_DIR}'char_dict.json' \
--ord_map_dict_path ${DATA_TARGET_DIR}'ord_map.json' \
--txt_path ${DATA_TARGET_DIR}'label/image_list_test.txt' \
--visualize 0