#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-2-26 下午9:03
# @Site    : https://github.com/MaybeShewill-CV/CRNN_Tensorflow
# @File    : shadownet_data_feed_pipline.py
# @IDE: PyCharm
"""
Synth90k dataset feed pipline
"""
import os
import os.path as ops
import random
import time

import glob
import glog as log
import tqdm
import tensorflow as tf

from config import model_config
from data_provider import tf_io_pipline_fast_tools

CFG = model_config.cfg


class CrnnDataProducer(object):
    """
    Convert raw image file into tfrecords
    """
    def __init__(self, dataset_dir, char_dict_path=None, ord_map_dict_path=None,
                 writer_process_nums=4):
        """
        init crnn data producer
        :param dataset_dir: image dataset root dir
        :param char_dict_path: char dict path
        :param ord_map_dict_path: ord map dict path
        :param writer_process_nums: the number of writer process
        """
        if not ops.exists(dataset_dir):
            raise ValueError('Dataset dir {:s} not exist'.format(dataset_dir))

        # Check image source data
        self._dataset_dir = dataset_dir
        self._train_annotation_file_path = ops.join(dataset_dir, 'label_train.txt')
        self._test_annotation_file_path = ops.join(dataset_dir, 'label_test.txt')
        self._val_annotation_file_path = ops.join(dataset_dir, 'label_val.txt')
        self._char_dict_path = char_dict_path
        self._ord_map_dict_path = ord_map_dict_path
        self._writer_process_nums = writer_process_nums

        if not self._is_source_data_complete():
            raise ValueError('Source image data is not complete, '
                             'please check if one of the image folder '
                             'or index file is not exist')

        # Init training example information
        self._train_sample_infos = []
        self._test_sample_infos = []
        self._val_sample_infos = []
        self._init_dataset_sample_info()

        # Check if need generate char dict map
        if char_dict_path is None or ord_map_dict_path is None:
            os.makedirs('./data/char_dict', exist_ok=True)
            self._char_dict_path = ops.join('./data/char_dict', 'char_dict.json')
            self._ord_map_dict_path = ops.join('./data/char_dict', 'ord_map.json')

    def generate_tfrecords(self, save_dir):
        """
        Generate tensorflow records file
        :param save_dir: tensorflow records save dir
        :return:
        """
        # make save dirs
        os.makedirs(save_dir, exist_ok=True)

        # generate training example tfrecords
        log.info('Generating training sample tfrecords...')
        t_start = time.time()

        tfrecords_writer = tf_io_pipline_fast_tools.CrnnFeatureWriter(
            annotation_infos=self._train_sample_infos,
            char_dict_path=self._char_dict_path,
            ord_map_dict_path=self._ord_map_dict_path,
            tfrecords_save_dir=save_dir,
            writer_process_nums=self._writer_process_nums,
            dataset_flag='train'
        )
        tfrecords_writer.run()

        log.info('Generate training sample tfrecords complete, cost time: {:.5f}'.format(time.time() - t_start))

        # generate val example tfrecords
        log.info('Generating validation sample tfrecords...')
        t_start = time.time()

        tfrecords_writer = tf_io_pipline_fast_tools.CrnnFeatureWriter(
            annotation_infos=self._val_sample_infos,
            char_dict_path=self._char_dict_path,
            ord_map_dict_path=self._ord_map_dict_path,
            tfrecords_save_dir=save_dir,
            writer_process_nums=self._writer_process_nums,
            dataset_flag='val'
        )
        tfrecords_writer.run()

        log.info('Generate validation sample tfrecords complete, cost time: {:.5f}'.format(time.time() - t_start))

        # generate test example tfrecords
        log.info('Generating testing sample tfrecords....')
        t_start = time.time()

        tfrecords_writer = tf_io_pipline_fast_tools.CrnnFeatureWriter(
            annotation_infos=self._test_sample_infos,
            char_dict_path=self._char_dict_path,
            ord_map_dict_path=self._ord_map_dict_path,
            tfrecords_save_dir=save_dir,
            writer_process_nums=self._writer_process_nums,
            dataset_flag='test'
        )
        tfrecords_writer.run()

        log.info('Generate testing sample tfrecords complete, cost time: {:.5f}'.format(time.time() - t_start))

        return

    def _is_source_data_complete(self):
        """
        Check if source data complete
        :return:
        """
        return \
            ops.exists(self._train_annotation_file_path) and ops.exists(self._val_annotation_file_path) \
            and ops.exists(self._test_annotation_file_path) 

    def _init_dataset_sample_info(self):
        """
        organize dataset sample information, read all the lexicon information in lexicon list.
        Train, test, val sample information are lists like
        [(image_absolute_path_1, image_lexicon_index_1), (image_absolute_path_2, image_lexicon_index_2), ...]
        :return:
        """
 
        # establish train example info
        log.info('Start initialize train sample information list...')
        num_lines = sum(1 for _ in open(self._train_annotation_file_path, 'r'))
        with open(self._train_annotation_file_path, 'r', encoding='utf-8') as file:
            for line in tqdm.tqdm(file, total=num_lines):

                image_name, label_index = line.rstrip('\r').rstrip('\n').split(' ')
                image_path = ops.join(self._dataset_dir, image_name)
                label_index = int(label_index)

                if not ops.exists(image_path):
                    raise ValueError('Example image {:s} not exist'.format(image_path))

                self._train_sample_infos.append((image_path, label_index))

        # establish val example info
        log.info('Start initialize validation sample information list...')
        num_lines = sum(1 for _ in open(self._val_annotation_file_path, 'r'))
        with open(self._val_annotation_file_path, 'r', encoding='utf-8') as file:
            for line in tqdm.tqdm(file, total=num_lines):
                image_name, label_index = line.rstrip('\r').rstrip('\n').split(' ')
                image_path = ops.join(self._dataset_dir, image_name)
                label_index = int(label_index)

                if not ops.exists(image_path):
                    raise ValueError('Example image {:s} not exist'.format(image_path))

                self._val_sample_infos.append((image_path, label_index))

        # establish test example info
        log.info('Start initialize testing sample information list...')
        num_lines = sum(1 for _ in open(self._test_annotation_file_path, 'r'))
        with open(self._test_annotation_file_path, 'r', encoding='utf-8') as file:
            for line in tqdm.tqdm(file, total=num_lines):
                image_name, label_index = line.rstrip('\r').rstrip('\n').split(' ')
                image_path = ops.join(self._dataset_dir, image_name)
                label_index = int(label_index)

                if not ops.exists(image_path):
                    raise ValueError('Example image {:s} not exist'.format(image_path))

                self._test_sample_infos.append((image_path, label_index))



class CrnnDataFeeder(object):
    """
    Read training examples from tfrecords for crnn model
    """
    def __init__(self, dataset_dir, char_dict_path, ord_map_dict_path, flags='train'):
        """
        crnn net dataset io pip line
        :param dataset_dir: the root dir of crnn dataset
        :param char_dict_path: json file path which contains the map relation
        between ord value and single character
        :param ord_map_dict_path: json file path which contains the map relation
        between int index value and char ord value
        :param flags: flag to determinate for whom the data feeder was used
        """
        self._dataset_dir = dataset_dir

        self._tfrecords_dir = ops.join(dataset_dir, 'tfrecords')
        if not ops.exists(self._tfrecords_dir):
            raise ValueError('{:s} not exist, please check again'.format(self._tfrecords_dir))

        self._dataset_flags = flags.lower()
        if self._dataset_flags not in ['train', 'test', 'valid']:
            raise ValueError('flags of the data feeder should be \'train\', \'test\', \'valid\'')

        #self._char_dict_path = char_dict_path
        #self._ord_map_dict_path = ord_map_dict_path
        #self._tfrecords_io_reader = tf_io_pipline_fast_tools.CrnnFeatureReader(
        #    char_dict_path=self._char_dict_path, ord_map_dict_path=self._ord_map_dict_path)
        #self._tfrecords_io_reader.dataset_flags = self._dataset_flags

    def sample_counts(self):
        """
        use tf records iter to count the total sample counts of all tfrecords file
        :return: int: sample nums
        """
        tfrecords_file_paths = glob.glob('{:s}/{:s}*.tfrecords'.format(self._tfrecords_dir, self._dataset_flags))
        counts = 0

        for record in tfrecords_file_paths:
            counts += sum(1 for _ in tf.python_io.tf_record_iterator(record))

        return counts

    @staticmethod
    def _augment_for_train(input_images, input_labels, input_image_paths):
        """

        :param input_images:
        :param input_labels:
        :param input_image_paths:
        :return:
        """
        return input_images, input_labels, input_image_paths

    @staticmethod
    def _augment_for_validation(input_images, input_labels, input_image_paths):
        """

        :param input_images:
        :param input_labels:
        :param input_image_paths:
        :return:
        """
        return input_images, input_labels, input_image_paths

    @staticmethod
    def _normalize(input_images, input_labels, input_image_paths):
        """

        :param input_images:
        :param input_labels:
        :param input_image_paths:
        :return:
        """
        input_images = tf.subtract(tf.divide(input_images, 127.5), 1.0)
        return input_images, input_labels, input_image_paths
    
    def inputs(self, batch_size):
        """
        Supply the batched data for training, testing and validation. For training and validation
        this function will run in a infinite loop until user end it outside of the function.
        For testing this function will raise an tf.errors.OutOfRangeError when reach the end of
        the dataset. User may catch this exception to terminate a loop.
        :param batch_size:
        :return: A tuple (images, labels, image_paths), where:
                    * images is a float tensor with shape [batch_size, H, W, C]
                      in the range [-1.0, 1.0].
                    * labels is an sparse tensor with shape [batch_size, None] with the true label
                    * image_paths is an tensor with shape [batch_size] with the image's absolute file path
        """
        print('================={:s}/{:s}*.tfrecords'.format(self._tfrecords_dir, self._dataset_flags))
        tfrecords_file_paths = glob.glob('{:s}/{:s}*.tfrecords'.format(self._tfrecords_dir, self._dataset_flags))

        if not tfrecords_file_paths:
            raise ValueError('Dataset does not contain any tfrecords for {:s}'.format(self._dataset_flags))

        random.shuffle(tfrecords_file_paths)

        return self._inputs(
            tfrecords_path=tfrecords_file_paths,
            batch_size=batch_size,
            num_threads=CFG.TRAIN.CPU_MULTI_PROCESS_NUMS
        )

    def _extract_features_batch(self, serialized_batch):
        features = tf.parse_example(
            serialized_batch,
            features={'images': tf.FixedLenFeature([], tf.string),
                'imagepaths': tf.FixedLenFeature([], tf.string),
                'labels': tf.VarLenFeature(tf.int64),
                 })

        bs = features['images'].shape[0]
        images = tf.decode_raw(features['images'], tf.uint8)
        w, h = tuple(CFG.ARCH.INPUT_SIZE)
        images = tf.cast(x=images, dtype=tf.float32)
        images = tf.subtract(tf.divide(images, 127.5), 1.0)
        #images = tf.reshape(images, [bs, h, -1, CFG.ARCH.INPUT_CHANNELS])
        images = tf.reshape(images, [bs, h, -1, CFG.ARCH.INPUT_CHANNELS])
        labels = features['labels']
        labels = tf.cast(labels, tf.int32)

        imagepaths = features['imagepaths']

        return images, labels, imagepaths


    def _inputs(self, tfrecords_path, batch_size, num_threads):
        dataset = tf.data.TFRecordDataset(tfrecords_path)
        dataset = dataset.batch(batch_size, drop_remainder=True)

        dataset = dataset.map(map_func=self._extract_features_batch, num_parallel_calls=num_threads)

        if self._dataset_flags != 'test':
            dataset = dataset.shuffle(buffer_size=1000)
            dataset = dataset.repeat()

        iterator = dataset.make_one_shot_iterator()

        return iterator.get_next(name='{:s}_IteratorGetNext'.format(self._dataset_flags))