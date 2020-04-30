#!/bin/bash



startTime=`date +%Y%m%d-%H:%M`
startTime_s=`date +%s`

OUTPUT_DIR='./output/'

RAW_DATA_DIR=${OUTPUT_DIR}'mjsynth_data/'

ZIP_DATA_PATH=${RAW_DATA_DIR}'mjsynth.tar.gz'

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
    echo "Download  http://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz"
    cd ${RAW_DATA_DIR}
    wget -c http://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz
    tar -xzvf mjsynth.tar.gz
    cd -
else
    echo  ${ZIP_DATA_PATH} "File already exists"
    exit
fi


endTime=`date +%Y%m%d-%H:%M`
endTime_s=`date +%s`
echo  "Done. Start Time --> End Time" "$startTime ---> $endTime"


