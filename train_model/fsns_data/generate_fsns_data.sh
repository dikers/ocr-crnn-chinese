echo $#  '生成中文数据集'
if [ $# -ne 3 ]
then
    echo "Usage: $0 '../../sample_data/test.txt'  'val_rate(0.2)'   'output_dir' "
    exit
fi

startTime=`date +%Y%m%d-%H:%M`
startTime_s=`date +%s`

DATA_TARGET_DIR=../../output/
BASE_DIR=$3

if [ ! -d ${DATA_TARGET_DIR} ];then
mkdir ${DATA_TARGET_DIR}
fi

echo 'input file line count: '
wc -l $1

export PYTHONPATH=./
python3 fsns_segment_string.py -mi 14 -ma 14 -i  $1  --output_dir ${DATA_TARGET_DIR}


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

if [ ! -d ${BASE_DIR}"data" ]
then
    mkdir ${BASE_DIR}"data"
    mkdir ${BASE_DIR}"data/train"
    mkdir ${BASE_DIR}"data/valid"
fi

trdg \
-c $train_count -l cn -i ${BASE_DIR}'train.txt' -na 2 \
--output_dir ${BASE_DIR}"data/train" -ft "../../sample_data/font/test01.ttf"

mv ${BASE_DIR}"data/train/labels.txt" ${BASE_DIR}"train_labels.txt"


trdg \
-c $val_count -l cn -i ${BASE_DIR}'valid.txt' -na 2 \
--output_dir ${BASE_DIR}"data/valid" -ft "../../sample_data/font/test01.ttf"

mv ${BASE_DIR}"data/valid/labels.txt" ${BASE_DIR}"valid_labels.txt"

python3 rename_label_file.py -i ${BASE_DIR}"train_labels.txt" -o ${BASE_DIR}"data/train"


python3 rename_label_file.py -i ${BASE_DIR}"valid_labels.txt" -o ${BASE_DIR}"data/valid"
