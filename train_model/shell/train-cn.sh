DATA_TARGET_DIR='./data_cn/'

export PYTHONPATH=../../
python3 ../tools/train_crnn.py \
--dataset_dir=${DATA_TARGET_DIR}'tfrecords/train/' \
-c=${DATA_TARGET_DIR}'char_map.json' \
-s=${DATA_TARGET_DIR}'model_save'
