# 生成labelme 格式数据
#
#
export PYTHONPATH=../

python ./generate_labelme_format.py \
--input_dir='../target/' \
--output_dir='../output/'