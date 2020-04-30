#   OCR 识别


通过调用trdg，自动生成中文手写体图片， 然后通过crnn+ctc进行文本识别。


## 建立环境

```shell script
conda create -n  ocr-cn python=3.6 pip scipy numpy ##运用conda 创建python环境
source activate ocr-cn
pip install -r requirements.txt -i https://mirrors.163.com/pypi/simple/
```


## 准备数据

```shell script
sh ./shell/get_sample_data.sh
```

## 英文字符数据集
```shell script
sh ./shell/get_mjsynth_data
```


##  生成tfrecord记录  

* 文本生成图片  [TRDG 文本生成图片代码](https://github.com/Belval/TextRecognitionDataGenerator)
* 可以添加多种手写字体文件  [免费中文字体文件下载地址](http://www.sucaijishi.com/material/font/)


```shell script
# 少量测试数据
sh ./shell/generation_cn_hw_tfrecord.sh   ./sample_data/test.txt  0.2  

# 中等测试数据  需要先执行（sh ./shell/get_sample_data.sh）
sh ./shell/generation_cn_hw_tfrecord.sh   ./output/raw_data/cnews.val.txt  0.2  

# 大量测试数据  需要先执行（sh ./shell/get_sample_data.sh）
sh ./shell/generation_cn_hw_tfrecord.sh   ./output/raw_data/cnews.train.txt  0.1  
```


## 生成数据

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
./output/tfrecords/valid  # 验证集
```





##  训练模型

```shell script
sh shell/train_crnn.py
```
[CRNN 参考代码 - CRNN_Tensorflow](https://github.com/MaybeShewill-CV/CRNN_Tensorflow)

[CRNN 参考代码 - crnn_ctc_ocr_tf](https://github.com/bai-shang/crnn_ctc_ocr_tf)




## 使用jupyter进行调试

[./notebook/train-cn.ipynb](./notebook/train-cn.ipynb)

##  测试模型

```shell script
sh shell/test_crnn.py
```



# 对比代码

[参考代码地址](https://github.com/Mingtzge/2019-CCF-BDCI-OCR-MCZJ-OCR-IdentificationIDElement)

操作步骤
```shell

cd recognize_process/shell

# 生成数据
sh generation_cn_tfrecord.sh  'YOUR_TXT_PATH'  0.2


#sh generation_cn_tfrecord.sh  ../../output/raw_data/cnews.val.txt 0.02  # 使用前一步准备好的数据

# 训练
sh train-cn.sh

# 查看测试结果
sh test-cn.sh
```

生成数据的路径在 './recognize_process/shell/data_cn/'