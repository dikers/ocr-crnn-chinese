#!/bin/bash



startTime=`date +%Y%m%d-%H:%M`
startTime_s=`date +%s`

OUTPUT_DIR='./output/'

RAW_DATA_DIR=${OUTPUT_DIR}'raw_data/'

ZIP_DATA_PATH=${RAW_DATA_DIR}'cnews_data.zip'

if [ ! -d ${OUTPUT_DIR} ]
then
    mkdir ${OUTPUT_DIR}
fi

if [ ! -d ${RAW_DATA_DIR} ]
then
    mkdir ${RAW_DATA_DIR}
fi


if [ ! -f ${ZIP_DATA_PATH} ]
then
    echo "The original file does not exist, Download started."
    echo "Download  https://dikers-data.s3.cn-northwest-1.amazonaws.com.cn/dataset/cnews_data.zip"
    cd ${RAW_DATA_DIR}
    wget https://dikers-data.s3.cn-northwest-1.amazonaws.com.cn/dataset/cnews_data.zip
    unzip  cnews_data.zip
    rm -fr __MACOSX
    cd -
else
    echo  ${ZIP_DATA_PATH} "File already exists"
    exit
fi


endTime=`date +%Y%m%d-%H:%M`
endTime_s=`date +%s`
echo  "Done. Start Time --> End Time" "$startTime ---> $endTime"


