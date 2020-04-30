#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-29 下午3:56
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/CRNN_Tensorflow
# @File    : test_shadownet.py
# @IDE: PyCharm Community Edition
"""
Use shadow net to recognize the scene text of a single image
"""
import argparse
import os.path as ops

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glog as logger
import wordninja
import json

from config import global_config
from crnn_model import crnn_net
from data_provider import tf_io_pipline_fast_tools

CFG = global_config.cfg


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"




def init_args():
    """

    :return: parsed arguments and (updated) config.cfg object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str,
                        help='Path to the image to be tested',
                        default='data/test_images/test_01.jpg')
    parser.add_argument('--weights_path', type=str,
                        help='Path to the pre-trained weights to use')
    parser.add_argument('-c', '--char_dict_path', type=str,
                        help='Directory where character dictionaries for the dataset were stored')
    parser.add_argument('-o', '--ord_map_dict_path', type=str,
                        help='Directory where ord map dictionaries for the dataset were stored')
    parser.add_argument('-v', '--visualize', type=args_str2bool, nargs='?', const=True,
                        help='Whether to display images')
    parser.add_argument('-t', '--txt_path', type=str,
                        help='test image label file')

    return parser.parse_args()


def get_num_class(char_dict_path):
    """
    get the number of char classes automatically
    :param char_dict_path: path for char_dictionary
    """
    char_map_dict = json.load(open(char_dict_path, 'r',encoding='utf-8'))
    if char_map_dict is None:
        print("error")
    assert (isinstance(char_map_dict, dict) and 'char_map_dict is not a dict')
    num_class = len(char_map_dict.keys())+1
    return num_class


def args_str2bool(arg_value):
    """

    :param arg_value:
    :return:
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def recognize(image_path, weights_path, char_dict_path, ord_map_dict_path, is_vis, is_english, txt_path):
    """

    :param image_path:
    :param weights_path:
    :param char_dict_path:
    :param ord_map_dict_path:
    :param is_vis:
    :param is_english:
    :return:
    """
    test_number=20
    print('Test file path {} '.format(txt_path) )
    NUM_CLASSES = get_num_class(char_dict_path)    
    print('num_classes: ',  NUM_CLASSES)
    
    with open(txt_path, 'r') as f1:
        linelist = f1.readlines()
    
    image_list = []   
    for i in range(test_number):
        image_path_temp = image_path + linelist[i].split(' ')[0]
        image_list.append((image_path_temp, linelist[i].split(' ')[1].replace('\r','').replace('\n','').replace('\t','')))


    inputdata = tf.placeholder(
        dtype=tf.float32,
        shape=[1, CFG.ARCH.INPUT_SIZE[1], CFG.ARCH.INPUT_SIZE[0], CFG.ARCH.INPUT_CHANNELS],
        name='input'
    )

    codec = tf_io_pipline_fast_tools.CrnnFeatureReader(
        char_dict_path=char_dict_path,
        ord_map_dict_path=ord_map_dict_path
    )

    net = crnn_net.ShadowNet(
        phase='test',
        hidden_nums=CFG.ARCH.HIDDEN_UNITS,
        layers_nums=CFG.ARCH.HIDDEN_LAYERS,
        num_classes=NUM_CLASSES
    )

    inference_ret = net.inference(
        inputdata=inputdata,
        name='shadow_net',
        reuse=False
    )

    decodes, _ = tf.nn.ctc_beam_search_decoder(
        inputs=inference_ret,
        sequence_length=int(CFG.ARCH.INPUT_SIZE[0] / 4) * np.ones(1),
        merge_repeated=False,
        beam_width=10
    )

    # config tf saver
    saver = tf.train.Saver()

    # config tf session
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TEST.TF_ALLOW_GROWTH

    sess = tf.Session(config=sess_config)
    
    weights_path = tf.train.latest_checkpoint(weights_path)
    print('weights_path: ', weights_path)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)
        for image_name, label in image_list:
            image = cv2.imread(image_name, cv2.IMREAD_COLOR)
            image = cv2.resize(image, dsize=tuple(CFG.ARCH.INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            image_vis = image
            image = np.array(image, np.float32) / 127.5 - 1.0
            
            preds = sess.run(decodes, feed_dict={inputdata: [image]})

            preds = codec.sparse_tensor_to_str(preds[0])[0]
            if is_english:
                preds = ' '.join(wordninja.split(preds))

            print('Label[{:20s}] Pred:[{:20s}]'.format(label, preds))

            if is_vis:
                plt.figure('CRNN Model Demo')
                plt.imshow(image_vis[:, :, (2, 1, 0)])
                plt.show()

    sess.close()

    return


if __name__ == '__main__':
    """
    
    """
    # init images
    args = init_args()
    
    # detect images
    recognize(
        image_path=args.image_path,
        weights_path=args.weights_path,
        char_dict_path=args.char_dict_path,
        ord_map_dict_path=args.ord_map_dict_path,
        is_vis=args.visualize,
        is_english=False,
        txt_path=args.txt_path
    )
