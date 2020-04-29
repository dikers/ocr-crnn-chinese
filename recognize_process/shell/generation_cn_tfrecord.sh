#!/bin/bash

echo $#  '生成中文数据集'
if [ $# -ne 2 ]
then
    echo "Usage: $0 '../../sample_data/test.txt'  'val_rate(0.2)' "
    exit
fi

startTime=`date +%Y%m%d-%H:%M`
startTime_s=`date +%s`

BASE_DIR="./data_cn/"

if [ ! -d ${BASE_DIR} ];then
mkdir ${BASE_DIR}
fi

export PYTHONPATH=../../
echo "start --------- segment  string -- "
python3 ../../train_model/data_provider/segment_string.py -mi 14 -ma 14 -i $1 --output_dir ${BASE_DIR}

echo 'input file line count: '
wc -l $1

echo "start --------- generate  image --"
TOTAL_COUNT=$(wc -l ${BASE_DIR}'/text_split.txt' | awk '{print $1}')

echo 'val_rate: ' $2 ' count ' ${TOTAL_COUNT}


val_count=`echo "scale=0; ${TOTAL_COUNT} * $2" | bc`
val_count=`echo $val_count | awk -F. '{print $1}'`

train_count=$[TOTAL_COUNT - val_count]

echo 'total  count: '  ${TOTAL_COUNT}
echo 'test   count: '  ${val_count}
echo 'train  count: '  ${train_count}

head -n ${train_count} ${BASE_DIR}'text_split.txt'  > ${BASE_DIR}'train.txt'
tail -n ${val_count}   ${BASE_DIR}'text_split.txt'  > ${BASE_DIR}'valid.txt'


if [ ! -d ${BASE_DIR}"tfrecords" ]
then
    mkdir ${BASE_DIR}"tfrecords"
fi

if [ ! -d ${BASE_DIR}"images" ]
then
    mkdir ${BASE_DIR}"images"
    mkdir ${BASE_DIR}"images/train"
    mkdir ${BASE_DIR}"images/valid"
fi
 
trdg \
-c $train_count -l cn -i ${BASE_DIR}'train.txt' -na 2 \
--output_dir ${BASE_DIR}"images/train" -ft "../../sample_data/font/test01.ttf"

mv ${BASE_DIR}"images/train/labels.txt" ${BASE_DIR}"train_labels.txt"
 
 
trdg \
-c $val_count -l cn -i ${BASE_DIR}'valid.txt' -na 2 \
--output_dir ${BASE_DIR}"images/valid" -ft "../../sample_data/font/test01.ttf"




mv ${BASE_DIR}"/images/valid/labels.txt" ${BASE_DIR}"valid_labels.txt"

head -n 20 ${BASE_DIR}"train_labels.txt"  > ${BASE_DIR}'test_labels.txt'
 
echo "start --------- generate  tfrecord "


python ../data_provider/write_tfrecord.py \
--dataset_dir=${BASE_DIR}'images/train' \
--char_dict_path=${BASE_DIR}'char_map.json' \
--anno_file_path=${BASE_DIR}'train_labels.txt' \
--save_dir=${BASE_DIR}'tfrecords/train/'


python ../data_provider/write_tfrecord.py \
--dataset_dir=${BASE_DIR}'images/valid' \
--char_dict_path=${BASE_DIR}'char_map.json' \
--anno_file_path=${BASE_DIR}'valid_labels.txt' \
--save_dir=${BASE_DIR}'tfrecords/valid/'





endTime=`date +%Y%m%d-%H:%M`
endTime_s=`date +%s`
echo "$startTime ---> $endTime"