#!/bin/bash

#!/usr/bin/env bash
if [ $# -ne 2 ]
then
    echo "Usage: $0 sample_count(1000)  'val_rate(0.2)' "
    exit
fi

startTime=`date +%Y%m%d-%H:%M`
startTime_s=`date +%s`

DATA_SRC_DIR='/home/ec2-user/workspace/crnn_ctc_ocr_tf/data/mnt/ramdisk/max/90kDICT32px/'
DATA_TARGET_DIR='./data_en/'


if [ ! -d ${DATA_TARGET_DIR} ];then
mkdir ${DATA_TARGET_DIR}
fi


export PYTHONPATH=../
rm -fr ${DATA_TARGET_DIR}'tfrecords/*'
#mkdir ${DATA_TARGET_DIR}'tfrecords'


 

val_count=`echo "scale=0; $1 * $2" | bc`
val_count=`echo $val_count | awk -F. '{print $1}'`
test_count=20

echo 'train  count: '  $1
echo 'val    count: '  $val_count
echo 'test   count: '  ${test_count}

echo '生成训练数据------------- start'
head -n $1 ${DATA_SRC_DIR}'annotation_train.txt' > ${DATA_SRC_DIR}'label_train.txt'
head -n $val_count ${DATA_SRC_DIR}'annotation_val.txt'   > ${DATA_SRC_DIR}'label_val.txt'
head -n ${test_count} ${DATA_SRC_DIR}'annotation_test.txt'  > ${DATA_SRC_DIR}'label_test.txt'


cp './char_dict_en.json'  ${DATA_TARGET_DIR}'char_dict.json'
cp './ord_map_en.json'  ${DATA_TARGET_DIR}'ord_map.json'

echo "start --------- generate  tfrecord "
python3 ../tools/write_tfrecords.py \
--dataset_dir ${DATA_SRC_DIR} \
--char_dict_path ${DATA_TARGET_DIR}'char_dict.json' \
--ord_map_dict_path ${DATA_TARGET_DIR}'ord_map.json' \
--save_dir=${DATA_TARGET_DIR}'tfrecords/'


endTime=`date +%Y%m%d-%H:%M`
endTime_s=`date +%s`
echo "$startTime ---> $endTime"