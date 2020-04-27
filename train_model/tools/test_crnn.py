##!/usr/bin/python
# -*- coding: utf-8 -*-
# @Software: PyCharm
"""
识别图片中的文本。需要的参数有：
    1.图片所在路径。
    2.模型训练使用的charmap
    3.加载模型的路径

输出结果为：
    识别的字段string
"""
import sys
import os
rootPath = os.path.dirname(sys.path[0])
sys.path.append(rootPath)
import argparse
import os
import cv2
import numpy as np
import tensorflow as tf
import json
import glog as logger
from train_model.config import model_config
from train_model.crnn_model import crnn_model
CFG = model_config.cfg




def _resize_image(img):
    """
    用于将图片resize为固定高度（32）
    :param img: 输入图片
    :return: resize为固定高度的图片
    """
    dst_height = CFG.ARCH.INPUT_SIZE[1]
    h_old, w_old, _ = img.shape
    height = dst_height
    width = int(w_old * height / h_old)
    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

    return resized_img


def _sparse_matrix_to_list(sparse_matrix, char_map_dict_path=None):
    """
    将矩阵拆分为list，参考：https://github.com/bai-shang/crnn_ctc_ocr.Tensorflow
    :param sparse_matrix:
    :param char_map_dict_path:
    :return:
    """
    indices = sparse_matrix.indices
    values = sparse_matrix.values
    dense_shape = sparse_matrix.dense_shape

    # the last index in sparse_matrix is ctc blanck note
    char_map_dict = json.load(open(char_map_dict_path, 'r',encoding='utf-8'))
    if char_map_dict is None:
        print("error")
    assert (isinstance(char_map_dict, dict) and 'char_map_dict is not a dict')

    dense_matrix = len(char_map_dict.keys()) * np.ones(dense_shape, dtype=np.int32)
    for i, indice in enumerate(indices):
        dense_matrix[indice[0], indice[1]] = values[i]
    string_list = []
    for row in dense_matrix:
        string = []
        for val in row:
            string.append(_int_to_string(val, char_map_dict))
        string_list.append(''.join(s for s in string if s != '*'))
    return string_list


def _int_to_string(value, char_map_dict=None):
    """
    将识别结果转化为string，参考：https://github.com/bai-shang/crnn_ctc_ocr.Tensorflow
    :param value:
    :param char_map_dict:
    :return:
    """
    if char_map_dict is None:
        print("error")
        #char_map_dict = json.load(open(FLAGS.char_map_json_file, 'r'))
    assert (isinstance(char_map_dict, dict) and 'char_map_dict is not a dict')

    for key in char_map_dict.keys():
        if char_map_dict[key] == int(value):
            return str(key)
        elif len(char_map_dict.keys()) == int(value):
            return ""
    raise ValueError('char map dict not has {:d} value. convert index to char failed.'.format(value))


def recognize_single_image(image_path, weights_path, char_dict_path):
    """
    识别函数
    :param image_path: 图片所在路径
    :param weights_path: 模型保存路径
    :param char_dict_path: 字典文件存放位置
    :return: None
    """
    tf.reset_default_graph()
    inputdata = tf.placeholder(dtype=tf.float32,
                               shape=[1, CFG.ARCH.INPUT_SIZE[1],
                                      None, CFG.ARCH.INPUT_CHANNELS],  # 宽度可变
                               name='input')
    input_sequence_length = tf.placeholder(tf.int32, shape=[1], name='input_sequence_length')
    net = crnn_model.ShadowNet(phase='test', hidden_nums=CFG.ARCH.HIDDEN_UNITS,
                               layers_nums=CFG.ARCH.HIDDEN_LAYERS, num_classes=CFG.ARCH.NUM_CLASSES)
    inference_ret = net.inference(inputdata=inputdata, name='shadow_net', reuse=False)
    decodes, _ = tf.nn.ctc_beam_search_decoder(inputs=inference_ret, sequence_length=input_sequence_length,  # 序列宽度可变
                                               merge_repeated=False, beam_width=1)

    # config tf saver
    saver = tf.train.Saver()

    # config tf session
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    # sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    # sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH

    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    weights_path = tf.train.latest_checkpoint(weights_path)
    print('Restore model from last model checkpoint {:s}'.format(weights_path))
    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(image_path+'is not exist')
        image_name=image_path.split('/')[-1]
        image = _resize_image(image)
        image = np.array(image, np.float32) / 127.5 - 1.0
        seq_len = np.array([image.shape[1] / 4], dtype=np.int32)
        preds = sess.run(decodes, feed_dict={inputdata: [image], input_sequence_length:seq_len})

        preds = _sparse_matrix_to_list(preds[0], char_dict_path)
        print('Predict image {:s} result: {:s}'.format(image_name, preds[0]))

    sess.close()

    return

if __name__ == '__main__':
    # init images
    image_path='output/images/train/0.jpg'
    weights_path='output/model_save/'
    char_dict_path='output/text_data/char_map.json'

    recognize_single_image(image_path,weights_path, char_dict_path)
