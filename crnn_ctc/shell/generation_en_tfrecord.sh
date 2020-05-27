#!/bin/bash

#!/usr/bin/env bash
if [ $# -ne 2 ]
then
    echo "Usage: $0 sample_count(1000)  'val_rate(0.2)' "
    echo "Usage: $0  10000   0.2' "
    exit
fi

startTime=`date +%Y%m%d-%H:%M`
startTime_s=`date +%s`


export PYTHONPATH=../../
# 需要需改自己的路径   下载地址 http://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz
DATA_SRC_DIR='../../output/mjsynth_data/mnt/ramdisk/max/90kDICT32px/'
DATA_TARGET_DIR='./data_en/'

if [ ! -d ${DATA_TARGET_DIR} ]
then
    echo '请先下载 mjsynth 训练数据集 地址如下: '
    echo 'http://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz'
    echo '保存路径：   output/mjsynth_data/mnt/ramdisk/max/90kDICT32px/ '
fi


if [ ! -d ${DATA_TARGET_DIR} ]
then
    mkdir ${DATA_TARGET_DIR}
    mkdir ${DATA_TARGET_DIR}'tfrecords'
else
    echo ${DATA_TARGET_DIR} "文件夹已经存在"
    rm -fr ${DATA_TARGET_DIR}'tfrecords'
fi



head -n $1 ${DATA_SRC_DIR}'annotation_train.txt' > ${DATA_TARGET_DIR}'image_list.txt'

echo "start --------- generate  image --"
count=$(wc -l ${DATA_TARGET_DIR}'image_list.txt' | awk '{print $1}')

echo 'val_rate: ' + $2

val_count=`echo "scale=0; $count * $2" | bc`
val_count=`echo $val_count | awk -F. '{print $1}'`

train_count=$[count - val_count]
test_count=20

echo 'total  count: ' + $count
echo 'val    count: '  $val_count
echo 'train  count: '  $train_count
echo 'test   count: '  ${test_count}


head ${DATA_TARGET_DIR}'image_list.txt' -n $train_count > ${DATA_TARGET_DIR}'image_list_train.txt'
tail ${DATA_TARGET_DIR}'image_list.txt' -n $val_count > ${DATA_TARGET_DIR}'image_list_valid.txt'


python3 ../../utils/change_label_text.py \
--input_file  ${DATA_TARGET_DIR}'image_list_train.txt' \
--output_dir ${DATA_TARGET_DIR}'label'

python3 ../../utils/change_label_text.py \
--input_file  ${DATA_TARGET_DIR}'image_list_valid.txt' \
--output_dir ${DATA_TARGET_DIR}'label'


head ${DATA_TARGET_DIR}'label/image_list_valid.txt' -n 20 > ${DATA_TARGET_DIR}'label/image_list_test.txt'

cp '../../sample_data/char_map_en.json'  ${DATA_TARGET_DIR}'char_map.json'
cp '../../sample_data/char_dict_en.json' ${DATA_TARGET_DIR}'char_dict.json'
cp '../../sample_data/ord_map_en.json'   ${DATA_TARGET_DIR}'ord_map.json'

echo "start --------- generate  tfrecord "
 
 
python ../data_provider/write_tfrecord.py \
--dataset_dir=${DATA_SRC_DIR} \
--char_dict_path=${DATA_TARGET_DIR}'char_map.json' \
--anno_file_path=${DATA_TARGET_DIR}'label/image_list_train.txt' \
--dataset_flag='train' \
--save_dir=${DATA_TARGET_DIR}'tfrecords'
 
python ../data_provider/write_tfrecord.py \
--dataset_dir=${DATA_SRC_DIR} \
--char_dict_path=${DATA_TARGET_DIR}'char_map.json' \
--anno_file_path=${DATA_TARGET_DIR}'label/image_list_valid.txt' \
--dataset_flag='valid' \
--save_dir=${DATA_TARGET_DIR}'tfrecords'

endTime=`date +%Y%m%d-%H:%M`
endTime_s=`date +%s`
echo "$startTime ---> $endTime"