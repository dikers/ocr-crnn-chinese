#!/bin/bash

if [ $# -ne 2 ]
then
    echo "Usage: $0 'input text file'  'val_rate(0.2)' "
    exit
fi

startTime=`date +%Y%m%d-%H:%M`
startTime_s=`date +%s`



OUTPUT_DIR='./output/'
RAW_DATA_DIR=${OUTPUT_DIR}'raw_data/'

ZIP_DATA_PATH=${RAW_DATA_DIR}'cnews_data.zip'

TEXT_DATA_DIR=${OUTPUT_DIR}'text_data/'
TFRECORDS_DATA_DIR=${OUTPUT_DIR}'tfrecords/'

IMAGES_DATA_DIR=${OUTPUT_DIR}'images/'

echo ${RAW_DATA_DIR}
if [ ! -d ${RAW_DATA_DIR} ]
then
    echo "Raw data does not exist."
    echo "Please run shell [get_sample_data.sh] first."
    exit
fi


if [ ! -f $1 ]
then
    echo   "File does not exist, Please confirm file  '$1' ."
    exit
fi

echo "Start parse '$1'  , valid_rate=$2 "


export PYTHONPATH=./
echo "start --------- segment  string -- "
python3 ./train_model/data_provider/segment_string.py  -mi 10 -ma 10 -i $1 --output_dir ${TEXT_DATA_DIR}


TEXT_SPLIT_FILE=${TEXT_DATA_DIR}'text_split.txt'
echo "start --------- generate  image --"
TOTAL_COUNT=$(wc -l ${TEXT_SPLIT_FILE} | awk '{print $1}')

echo 'val_rate: ' $2

val_count=`echo "scale=0; ${TOTAL_COUNT} * $2" | bc`
val_count=`echo $val_count | awk -F. '{print $1}'`

train_count=$[TOTAL_COUNT - val_count]

echo 'Total  count: '  ${TOTAL_COUNT}
echo 'Train  count: '  $train_count
echo 'Valid  count: '  $val_count

echo ${TEXT_SPLIT_FILE}
head -n ${train_count} ${TEXT_SPLIT_FILE}   > ${TEXT_DATA_DIR}'train.txt'
tail -n ${val_count} ${TEXT_SPLIT_FILE}  > ${TEXT_DATA_DIR}'valid.txt'
#
#
if [ ! -d ${TFRECORDS_DATA_DIR} ]
then
    mkdir ${TFRECORDS_DATA_DIR}
fi

rm -fr  ${TFRECORDS_DATA_DIR}'*'
mkdir ${TFRECORDS_DATA_DIR}"train"
mkdir ${TFRECORDS_DATA_DIR}"valid"



if [ ! -d ${IMAGES_DATA_DIR} ]
then
    mkdir ${IMAGES_DATA_DIR}
fi

rm -fr  ${IMAGES_DATA_DIR}'*'
mkdir ${IMAGES_DATA_DIR}"train"
mkdir ${IMAGES_DATA_DIR}"valid"


trdg \
-c $train_count -l cn -i ${TEXT_DATA_DIR}'train.txt' -na 2 \
--output_dir ${IMAGES_DATA_DIR}'train' -ft "./sample_data/font/test01.ttf"

trdg \
-c $val_count -l cn -i ${TEXT_DATA_DIR}'valid.txt' -na 2 \
--output_dir ${IMAGES_DATA_DIR}'valid' -ft "./sample_data/font/test01.ttf"

mv ${IMAGES_DATA_DIR}"train/labels.txt" ${TEXT_DATA_DIR}"train_labels.txt"
mv ${IMAGES_DATA_DIR}"valid/labels.txt" ${TEXT_DATA_DIR}"valid_labels.txt"



echo "start --------- generate  tfrecord "
python ./train_model/data_provider/write_tfrecord.py \
--dataset_dir=${IMAGES_DATA_DIR}"train" \
--char_dict_path=${TEXT_DATA_DIR}'char_map.json' \
--anno_file_path=${TEXT_DATA_DIR}'train_labels.txt' \
--dataset_flag='train' \
--save_dir=${TFRECORDS_DATA_DIR}"train"


python ./train_model/data_provider/write_tfrecord.py \
--dataset_dir=${IMAGES_DATA_DIR}"valid" \
--char_dict_path=${TEXT_DATA_DIR}'char_map.json' \
--anno_file_path=${TEXT_DATA_DIR}'valid_labels.txt' \
--dataset_flag='valid' \
--save_dir=${TFRECORDS_DATA_DIR}"valid"


endTime=`date +%Y%m%d-%H:%M`
endTime_s=`date +%s`
echo "$startTime ---> $endTime"