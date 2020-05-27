#   OCR 识别


通过调用trdg，自动生成中文手写体图片， 然后通过crnn+ctc进行文本识别。


## 建立环境

```shell script
conda create -n  ocr-cn python=3.6 pip scipy numpy ##运用conda 创建python环境
source activate ocr-cn
pip install -r requirements.txt -i https://mirrors.163.com/pypi/simple/
```


## 数据下载
```shell script
# 英文数据集
sh ./shell/get_mjsynth_data.sh
# 中文文本数据
sh ./shell/get_sample_data.sh
```

## 数据保存路径

生成数据在./output 下面

```shell script
.
├── mjsynth_data
│   ├── mjsynth.tar.gz
│   └── mnt                 #解压以后路径
└── raw_data
    ├── cnews_data.zip
    ├── cnews.test.txt
    ├── cnews.train.txt
    └── cnews.val.txt       #验证数据集 

```


##  训练脚本说明  

```shell script
cd ./crnn_ctc
tree -L 1

├── data_cn                         #中文训练临时文件夹
├── data_en                         #英文训练临时文件夹
├── generation_cn_tfrecord.sh       #生成中文Tfrecord 记录
├── generation_en_tfrecord.sh       #生成英文Tfrecord 记录
├── test-cn.sh                      #测试中文模型脚本
├── test-en.sh                      #测试英文模型脚本
├── train-cn.sh                     #训练中文模型
└── train-en.sh                     #训练英文模型

```

## 文字生成图片 

* 文本生成图片  [TRDG 文本生成图片代码](https://github.com/Belval/TextRecognitionDataGenerator)
* 可以添加多种手写字体文件  [免费中文字体文件下载地址](http://www.sucaijishi.com/material/font/)





##  训练模型

```shell script

cd ./crnn_ctc

sh generation_en_tfrecord.sh   10000  0.2    # 生成英文数据

sh generation_cn_tfrecord.sh  ../../sample_data/test.txt  0.2    #很少的数据， 用于验证环境是否正常

sh generation_cn_tfrecord.sh  ../../output/raw_data/cnews.val.txt 0.02  # 使用前一步准备好的数据

# 进行训练
sh train-cn.sh
sh train-en.sh

# 测试模型
sh test-cn.sh
sh test-en.sh


cd ./data_cn
tree -L 1
.
├── char_dict.json
├── char_map.json
├── images                  # 生成的图片
├── model_save              # 模型保存地址
├── ord_map.json            
├── test_labels.txt
├── text_split.txt
├── tfrecords               # tfrecords 路径
├── train_labels.txt
├── train.txt
├── valid_labels.txt
└── valid.txt

```

 
[CRNN 参考代码 - CRNN_Tensorflow](https://github.com/MaybeShewill-CV/CRNN_Tensorflow)

[CRNN 参考代码 - crnn_ctc_ocr_tf](https://github.com/bai-shang/crnn_ctc_ocr_tf)

[CRNN 参考代码  OCR IdentificationIDElement](https://github.com/Mingtzge/2019-CCF-BDCI-OCR-MCZJ-OCR-IdentificationIDElement)


## 使用jupyter进行调试

[./notebook/train-cn.ipynb](./notebook/train-cn.ipynb)




# 调用Baidu ocr自动生成训练样本


[百度OCR api 接口](https://ai.baidu.com/ai-doc/OCR/)


申请 开发ID 和Key
```python
from aip import AipOcr

""" 你的 APPID AK SK """
APP_ID = '你的 App ID'
API_KEY = '你的 Api Key'
SECRET_KEY = '你的 Secret Key'

client = AipOcr(APP_ID, API_KEY, SECRET_KEY)
```


### 修改 label_tools/ocr.sh

```shell script

export PYTHONPATH=./

python ./baidu_ocr.py \
--input_dir='../temp/' \
--output_dir='../target/' \
--app_id='app_id' \
--api_key='api_key' \
--secret_key='secret_key'
```


### 生成数据示例样本
```

.
├── test001_pdf_a98aac73    第一个PDF
│   ├── data.json           ocr 返回结果
│   ├── image.png           样本图片
│   ├── images              裁剪后的小图
│   └── labels.txt          标记文本数据
└── test002_pdf_6df2f695
    ├── data.json
    ├── image.png
    ├── images
    └── labels.txt

```


### 脚本执行过程

```shell script

cd  label_tools
sh ./ocr.sh
sh ./merge.sh
sh ./generation_tfrecord.sh 0.2

```
