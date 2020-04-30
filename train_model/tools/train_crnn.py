#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Reference    :  https://github.com/MaybeShewill-CV/CRNN_Tensorflow
#                  https://github.com/bai-shang/crnn_ctc_ocr.Tensorflow
#                  https://github.com/balancap/SSD-Tensorflow
# @File    : train_crnn.py
# @IDE: PyCharm Community Edition

import sys
sys.path.append('./')

import os
import os.path as ops
import time
import argparse
import math
import json
import glog as logger

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops

from train_model.tools import evaluation_tools
from train_model.crnn_model import crnn_model
from train_model.config import model_config
from train_model.data_provider import read_tfrecord

CFG = model_config.cfg


def init_args():
    """
    参数初始化
    :return: None
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-train', '--train_dataset_dir', type=str,
                        help='Directory containing train_features.tfrecords')
    parser.add_argument('-val', '--val_dataset_dir', type=str,
                        help='Directory containing train_features.tfrecords')
    parser.add_argument('-w', '--weights_path', type=str,
                        help='Path to pre-trained weights to continue training')
    parser.add_argument('-s', '--save_path', type=str,
                        help='Path to logs and ckpt models')
    parser.add_argument('-c', '--char_dict_path', type=str,
                        help='Directory where character dictionaries for the dataset were stored',
                        default='./step_three_recognize_process/char_map/char_map.json')
    parser.add_argument('-m', '--multi_gpus', type=int,
                        help='whether to use multi gpu to train the model, 1 for multi, 0 for single gpu',
                        default=1)

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

##########################################################################
def apply_with_random_selector(x, func, num_cases):
    """
    随机选择数据增强方式，参考：
    https://github.com/balancap/SSD-Tensorflow
    :param x:
    :param func:
    :param num_cases:
    :return:
    """
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
            func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
            for case in range(num_cases)])[0]


def distort_color(image, color_ordering=0, scope=None):
    """
    随机进行图像增强（亮度、对比度操作）
    :param image: 输入图片
    :param color_ordering:模式
    :param scope: 命名空间
    :return: 增强后的图片
    """
    with tf.name_scope(scope, 'distort_color', [image]):
        if color_ordering == 0:  # 模式0.先调整亮度，再调整对比度
            rand_temp = random_ops.random_uniform([], -55, 20, seed=None) # [-70, 30] for generate img, [-50, 20] for true img 
            image = math_ops.add(image, math_ops.cast(rand_temp, dtypes.float32))
            image = tf.image.random_contrast(image, lower=0.45, upper=1.5) # [0.3, 1.75] for generate img, [0.45, 1.5] for true img 
        else:
            image = tf.image.random_contrast(image, lower=0.45, upper=1.5)
            rand_temp = random_ops.random_uniform([], -55, 30, seed=None)
            image = math_ops.add(image, math_ops.cast(rand_temp, dtypes.float32))

        # The random_* ops do not necessarily clamp.
        print(color_ordering)
        return tf.clip_by_value(image, 0.0, 255.0)  # 限定在0-255
##########################################################################
def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads

def compute_net_gradients(images, labels, net, optimizer=None, is_net_first_initialized=False):
    """
    Calculate gradients for single GPU
    :param images: images for training
    :param labels: labels corresponding to images
    :param net: classification model
    :param optimizer: network optimizer
    :param is_net_first_initialized: if the network is initialized
    :return:
    """
    _, net_loss = net.compute_loss(
        inputdata=images,
        labels=labels,
        name='shadow_net',
        reuse=is_net_first_initialized
    )

    if optimizer is not None:
        grads = optimizer.compute_gradients(net_loss)
    else:
        grads = None

    return net_loss, grads

def train_shadownet_multi_gpu(dataset_dir_train, dataset_dir_val, weights_path, char_dict_path, model_save_dir):
    """

    :param dataset_dir:
    :param weights_path:
    :param char_dict_path:
    :param ord_map_dict_path:
    :return:
    """
    #caculdate num_class

    NUM_CLASSES = get_num_class(char_dict_path)
    train_dataset = read_tfrecord.CrnnDataFeeder(
        dataset_dir=dataset_dir_train,
        char_dict_path=char_dict_path,
        flags='train')

    val_dataset = read_tfrecord.CrnnDataFeeder(
        dataset_dir=dataset_dir_val,
        char_dict_path=char_dict_path,
        flags='valid')

    train_images, train_labels, train_images_paths = train_dataset.inputs(
        batch_size=CFG.TRAIN.BATCH_SIZE
    )
    val_images, val_labels, val_images_paths = val_dataset.inputs(
        batch_size=CFG.TRAIN.BATCH_SIZE)

    # set crnn net
    shadownet = crnn_model.ShadowNet(
        phase='train',
        hidden_nums=CFG.ARCH.HIDDEN_UNITS,
        layers_nums=CFG.ARCH.HIDDEN_LAYERS,
        num_classes=NUM_CLASSES
    )
    shadownet_val = crnn_model.ShadowNet(
        phase='test',
        hidden_nums=CFG.ARCH.HIDDEN_UNITS,
        layers_nums=CFG.ARCH.HIDDEN_LAYERS,
        num_classes=NUM_CLASSES
    )


    # set average container
    tower_grads = []
    train_tower_loss = []
    val_tower_loss = []
    batchnorm_updates = None
    train_summary_op_updates = None

    # set lr
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(
        learning_rate=CFG.TRAIN.LEARNING_RATE,
        global_step=global_step,
        decay_steps=CFG.TRAIN.LR_DECAY_STEPS,
        decay_rate=CFG.TRAIN.LR_DECAY_RATE,
        staircase=True)

    # set up optimizer
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

    # set distributed train op
    with tf.variable_scope(tf.get_variable_scope()):
        is_network_initialized = False
        for i in range(CFG.TRAIN.GPU_NUM):
            with tf.device('/gpu:{:d}'.format(i)):
                with tf.name_scope('tower_{:d}'.format(i)) as _:
                    train_loss, grads = compute_net_gradients(
                        train_images, train_labels, shadownet, optimizer,
                        is_net_first_initialized=is_network_initialized)

                    is_network_initialized = True

                    # Only use the mean and var in the first gpu tower to update the parameter
                    if i == 0:
                        batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                        train_summary_op_updates = tf.get_collection(tf.GraphKeys.SUMMARIES)

                    tower_grads.append(grads)
                    train_tower_loss.append(train_loss)
                with tf.name_scope('validation_{:d}'.format(i)) as _:
                    val_loss, _ = compute_net_gradients(
                        val_images, val_labels, shadownet_val, optimizer,
                        is_net_first_initialized=is_network_initialized)
                    val_tower_loss.append(val_loss)

    grads = average_gradients(tower_grads)
    avg_train_loss = tf.reduce_mean(train_tower_loss)
    avg_val_loss = tf.reduce_mean(val_tower_loss)

    # Track the moving averages of all trainable variables
    variable_averages = tf.train.ExponentialMovingAverage(
        CFG.TRAIN.MOVING_AVERAGE_DECAY, num_updates=global_step)
    variables_to_average = tf.trainable_variables() + tf.moving_average_variables()
    variables_averages_op = variable_averages.apply(variables_to_average)

    # Group all the op needed for training
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
    train_op = tf.group(apply_gradient_op, variables_averages_op,
                        batchnorm_updates_op)

    # set tensorflow summary
    tboard_save_path = model_save_dir
    os.makedirs(model_save_dir, exist_ok=True)

    summary_writer = tf.summary.FileWriter(tboard_save_path)

    avg_train_loss_scalar = tf.summary.scalar(name='average_train_loss',
                                              tensor=avg_train_loss)
    avg_val_loss_scalar = tf.summary.scalar(name='average_val_loss',
                                            tensor=avg_val_loss)
    learning_rate_scalar = tf.summary.scalar(name='learning_rate_scalar',
                                             tensor=learning_rate)
    train_merge_summary_op = tf.summary.merge(
        [avg_train_loss_scalar, learning_rate_scalar] + train_summary_op_updates
    )
    val_merge_summary_op = tf.summary.merge([avg_val_loss_scalar])

    # set tensorflow saver
    saver = tf.train.Saver()
    os.makedirs(model_save_dir, exist_ok=True)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'shadownet_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = ops.join(model_save_dir, model_name)

    # set sess config
    sess_config = tf.ConfigProto(device_count={'GPU': CFG.TRAIN.GPU_NUM}, allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    # Set the training parameters
    train_epochs = CFG.TRAIN.EPOCHS

    logger.info('Global configuration is as follows:')
    logger.info(CFG)

    sess = tf.Session(config=sess_config)

    summary_writer.add_graph(sess.graph)

    with sess.as_default():
        epoch = 0
        tf.train.write_graph(graph_or_graph_def=sess.graph, logdir='',
                             name='{:s}/shadownet_model.pb'.format(model_save_dir))

        if weights_path is None or not os.path.exists(weights_path) or len(os.listdir(weights_path)) < 5:
            logger.info('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            weights_path = tf.train.latest_checkpoint(weights_path)
            logger.info('Restore model from last model checkpoint {:s}'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)
            epoch = sess.run(tf.train.get_global_step())

        train_cost_time_mean = []
        val_cost_time_mean = []

        while epoch < train_epochs:
            epoch += 1
            # training part
            t_start = time.time()

            _, train_loss_value, train_summary, lr = \
                sess.run(fetches=[train_op,
                                  avg_train_loss,
                                  train_merge_summary_op,
                                  learning_rate])

            if math.isnan(train_loss_value):
                raise ValueError('Train loss is nan')

            cost_time = time.time() - t_start
            train_cost_time_mean.append(cost_time)

            summary_writer.add_summary(summary=train_summary,
                                       global_step=epoch)



            if epoch % CFG.TRAIN.DISPLAY_STEP == 0:
                logger.info('Epoch_Train: {:d} total_loss= {:6f} '
                            'lr= {:6f} mean_cost_time= {:5f}s '.
                            format(epoch + 1,
                                   train_loss_value,
                                   lr,
                                   np.mean(train_cost_time_mean)
                                   ))
                train_cost_time_mean.clear()

            if epoch % CFG.TRAIN.VAL_DISPLAY_STEP == 0:
                # validation part
                t_start_val = time.time()

                val_loss_value, val_summary = \
                    sess.run(fetches=[avg_val_loss,
                                      val_merge_summary_op])

                summary_writer.add_summary(val_summary, global_step=epoch)

                cost_time_val = time.time() - t_start_val
                val_cost_time_mean.append(cost_time_val)
                logger.info('Epoch_Val: {:d} total_loss= {:6f} '
                            ' mean_cost_time= {:5f}s '.
                            format(epoch + 1,
                                   val_loss_value,
                                   np.mean(val_cost_time_mean)))
                val_cost_time_mean.clear()

            if epoch % CFG.TRAIN.VAL_DISPLAY_STEP == 0:
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
    sess.close()

    return


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

def train_shadownet(dataset_dir_train,dataset_dir_val, weights_path, char_dict_path, model_save_dir):
    """
    训练网络，参考：
    https://github.com/MaybeShewill-CV/CRNN_Tensorflow
    :param dataset_dir: tfrecord文件路径
    :param weights_path: 要加载的预训练模型路径
    :param char_dict_path: 字典文件路径
    :param save_path: 模型保存路径
    :return: None
    """
    # prepare dataset
    train_dataset = read_tfrecord.CrnnDataFeeder(
        dataset_dir=dataset_dir_train, char_dict_path=char_dict_path, flags='train')

    train_images, train_labels, train_images_paths = train_dataset.inputs(
        batch_size=CFG.TRAIN.BATCH_SIZE)

####################添加数据增强##############################
    # train_images = tf.multiply(tf.add(train_images, 1.0), 128.0)   # removed since read_tfrecord.py is changed
    tf.summary.image('original_image', train_images)   # 保存到log，方便测试观察
    images = apply_with_random_selector(
        train_images,
        lambda x, ordering: distort_color(x, ordering),
        num_cases=2)  #
    images = tf.subtract(tf.divide(images, 127.5), 1.0)  # 转化到【-1，1】 changed 128.0 to 127.5 
    train_images = tf.clip_by_value(images, -1.0, 1.0)
    tf.summary.image('distord_turned_image', train_images)
################################################################

    NUM_CLASSES = get_num_class(char_dict_path)

    # declare crnn net
    shadownet = crnn_model.ShadowNet(phase='train',hidden_nums=CFG.ARCH.HIDDEN_UNITS,
        layers_nums=CFG.ARCH.HIDDEN_LAYERS, num_classes=NUM_CLASSES)
    
    # set up training graph
    with tf.device('/gpu:0'):
        # compute loss and seq distance
        train_inference_ret, train_ctc_loss = shadownet.compute_loss(inputdata=train_images,
            labels=train_labels, name='shadow_net', reuse=False)

        # set learning rate
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate=CFG.TRAIN.LEARNING_RATE,
            global_step=global_step, decay_steps=CFG.TRAIN.LR_DECAY_STEPS,
            decay_rate=CFG.TRAIN.LR_DECAY_RATE, staircase=True)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
            #    momentum=0.9).minimize(loss=train_ctc_loss, global_step=global_step)
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=\
                learning_rate).minimize(loss=train_ctc_loss, global_step=global_step)
            # 源代码优化器是momentum，改成adadelta，与CRNN论文一致


    # Set tf summary
    os.makedirs(save_path, exist_ok=True)
    tf.summary.scalar(name='train_ctc_loss', tensor=train_ctc_loss)
    tf.summary.scalar(name='learning_rate',  tensor=learning_rate)
    merge_summary_op = tf.summary.merge_all()

    # Set saver configuration
    saver = tf.train.Saver()
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'shadownet_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = ops.join(model_save_dir, model_name)

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess = tf.Session(config=sess_config)

    summary_writer = tf.summary.FileWriter(model_save_dir)
    summary_writer.add_graph(sess.graph)

    # Set the training parameters
    train_epochs = CFG.TRAIN.EPOCHS

    with sess.as_default():
        epoch = 0
        if weights_path is None:
            print('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            weights_path = tf.train.latest_checkpoint(weights_path)
            print('Restore model from last model checkpoint {:s}'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)
            epoch = sess.run(tf.train.get_global_step())

        cost_history = [np.inf]
        while epoch < train_epochs:
            epoch += 1
            _, train_ctc_loss_value, merge_summary_value, learning_rate_value = sess.run(
                [optimizer, train_ctc_loss, merge_summary_op, learning_rate])

            if (epoch+1) % CFG.TRAIN.DISPLAY_STEP == 0:
                
                current_time = time.strftime('%m-%d-%H-%M-%S', time.localtime(time.time()))
                print('{} lr={:.5f}  step:{:6d}   train_loss={:.4f}'.format(\
                    current_time, learning_rate_value, epoch+1, train_ctc_loss_value))
                
                # record history train ctc loss
                cost_history.append(train_ctc_loss_value)
                 # add training sumary
                summary_writer.add_summary(summary=merge_summary_value, global_step=epoch)

            if (epoch+1) % CFG.TRAIN.SAVE_STEPS == 0:
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)

    return np.array(cost_history[1:])  # Don't return the first np.inf


if __name__ == '__main__':
    # init args
    args = init_args()
    logger.info('start')
    if args.multi_gpus:
        logger.info('**************** Use multi gpus to train the model')
        train_shadownet_multi_gpu(
            dataset_dir_train=args.train_dataset_dir,
            dataset_dir_val=args.val_dataset_dir,
            weights_path=args.weights_path,
            char_dict_path=args.char_dict_path,
            model_save_dir=args.save_path
        )
    else:
        logger.info('***************** Use single gpu to train the model')
        train_shadownet(
            dataset_dir_train=args.train_dataset_dir,
            dataset_dir_val=args.val_dataset_dir,
            weights_path=args.weights_path,
            char_dict_path=args.char_dict_path,
            model_save_dir=args.save_path
        )
