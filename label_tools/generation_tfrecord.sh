#!/bin/bash

echo $#  '生成中文数据集'
if [ $# -ne 1 ]
then
    echo "Usage: $0  'val_rate(0.2)' "
    exit
fi

startTime=`date +%Y%m%d-%H:%M`
startTime_s=`date +%s`

BASE_DIR="../output/"

if [ ! -d ${BASE_DIR} ];then
mkdir ${BASE_DIR}
fi

export PYTHONPATH=../

echo "start --------- generate  image --"
TOTAL_COUNT=$(wc -l ${BASE_DIR}'labels.txt' | awk '{print $1}')

echo 'val_rate: ' $1 ' count ' ${TOTAL_COUNT}


val_count=`echo "scale=0; ${TOTAL_COUNT} * $1" | bc`
val_count=`echo $val_count | awk -F. '{print $1}'`

train_count=$[TOTAL_COUNT - val_count]

echo 'total  count: '  ${TOTAL_COUNT}
echo 'test   count: '  ${val_count}
echo 'train  count: '  ${train_count}

head -n ${train_count} ${BASE_DIR}'labels.txt'  > ${BASE_DIR}'train.txt'
tail -n ${val_count}   ${BASE_DIR}'labels.txt'  > ${BASE_DIR}'valid.txt'


if [ ! -d ${BASE_DIR}"tfrecords" ]
then
    mkdir ${BASE_DIR}"tfrecords"
else
    cd ${BASE_DIR}"tfrecords"
    rm -fr *
    cd -
fi

 
echo "start --------- generate  tfrecord "


python ../train_model/data_provider/write_tfrecord.py \
--dataset_dir=${BASE_DIR}'images' \
--char_dict_path=${BASE_DIR}'char_map.json' \
--anno_file_path=${BASE_DIR}'train.txt' \
--dataset_flag='train' \
--save_dir=${BASE_DIR}'tfrecords/train/'


python ../train_model/data_provider/write_tfrecord.py \
--dataset_dir=${BASE_DIR}'images' \
--char_dict_path=${BASE_DIR}'char_map.json' \
--anno_file_path=${BASE_DIR}'valid.txt' \
--dataset_flag='valid' \
--save_dir=${BASE_DIR}'tfrecords/valid/'




endTime=`date +%Y%m%d-%H:%M`
endTime_s=`date +%s`
echo "$startTime ---> $endTime"