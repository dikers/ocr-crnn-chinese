#   OCR 识别


### set up environment

```shell script
conda create -n  id-ocr python=3.6 pip scipy numpy ##运用conda 创建python环境
source activate id-ocr
pip install -r requirements.txt -i https://mirrors.163.com/pypi/simple/
```


### 下载数据

```shell script
sh ./shell/get_sample_data.sh
```


###  生成tfrecord 记录  
```shell script
# 少量测试数据
sh ./shell/generation_cn_hw_tfrecord.sh   ./sample_data/test.txt  0.2  

# 中等测试数据
sh ./shell/generation_cn_hw_tfrecord.sh   ./output/raw_data/cnews.val.txt  0.2  

# 大量测试数据
sh ./shell/generation_cn_hw_tfrecord.sh   ./output/raw_data/cnews.train.txt  0.1  
```


### 生成数据

生成数据在./output 下面

```shell script
.
├── images                    #文本生成的图片
│   ├── train
│   └── valid
├── raw_data                  #用来生成图片的文本数据
│   ├── cnews.test.txt
│   ├── cnews.train.txt
│   ├── cnews.val.txt
│   ├── cnews_data.zip
├── text_data
│   ├── char_map.json         # char_map
│   ├── text_split.txt
│   ├── train.txt
│   ├── train_labels.txt      # train labels
│   ├── valid.txt
│   └── valid_labels.txt      # valid  labels 
└── tfrecords
    ├── train                 # train tfrecord
    └── valid                 # valid tfrecord

```

*  训练数据路径

```shell script
./output/text_data/char_map.json   
./output/tfrecords/train  # 测试集
./output/tfrecords/valid  #  验证集
```



