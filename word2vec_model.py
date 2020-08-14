# -*- coding: utf-8 -*-
# @Time        : 2020/5/24 21:23
# @Author      : tianyunzqs
# @Description : 

import os
import math
import random
import logging
import pickle
import numpy as np
import tensorflow as tf


class TFWord2Vec(object):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 num_sampled,
                 lr):
        self.train_inputs = tf.placeholder(tf.int32, shape=[None], name='train_inputs')
        self.train_labels = tf.placeholder(tf.int32, shape=[None, 1], name='train_labels')
        self.embedding_dict = tf.Variable(
            tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0)
        )

        with tf.name_scope('loss'):
            self.nce_weight = tf.Variable(tf.truncated_normal([vocab_size, embedding_size],
                                                              stddev=1.0 / math.sqrt(embedding_size)))
            self.nce_biases = tf.Variable(tf.zeros([vocab_size]))

            # 将输入序列向量化
            embed = tf.nn.embedding_lookup(self.embedding_dict, self.train_inputs)  # batch_size
            # 得到NCE损失
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=self.nce_weight,
                    biases=self.nce_biases,
                    labels=self.train_labels,
                    inputs=embed,
                    num_sampled=num_sampled,
                    num_classes=vocab_size
                )
            )
            tf.summary.scalar('loss', self.loss)  # 让tensorflow记录参数

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(lr)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        # 计算与指定若干单词的相似度
        self.test_word_id = tf.placeholder(tf.int32, shape=[None], name='test_word_id')

        with tf.name_scope('output_vector'):
            self.test_word_vector = tf.nn.embedding_lookup(self.embedding_dict, self.test_word_id)

        with tf.name_scope('similarity'):
            vec_l2_model = tf.sqrt(  # 求各词向量的L2模
                tf.reduce_sum(tf.square(self.embedding_dict), 1, keep_dims=True)
            )
            avg_l2_model = tf.reduce_mean(vec_l2_model)
            tf.summary.scalar('avg_vec_model', avg_l2_model)
            self.normed_embedding = self.embedding_dict / vec_l2_model
            # self.embedding_dict = norm_vec # 对embedding向量正则化
            test_embed = tf.nn.embedding_lookup(self.normed_embedding, self.test_word_id)
            self.similarity = tf.matmul(test_embed, self.normed_embedding, transpose_b=True)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        self.merge_summary = tf.summary.merge_all()


class BatchManager(object):
    def __init__(self, data,  batch_size):
        self.batch_data = self.gen_batch(data, batch_size)
        self.len_data = len(self.batch_data)

    @staticmethod
    def gen_batch(data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(data[i*batch_size: (i+1)*batch_size])
        return batch_data

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]


class Word2Vec(object):
    def __init__(self,
                 sentences=None,
                 size=64,
                 window=3,
                 num_sampled=30,
                 lr=0.01,
                 epoch=5,
                 logger=None,
                 model_path=None,
                 is_training=True
                 ):
        self.sentences = sentences
        self.size = size
        self.window = window
        if not logger:
            self.logger = logging.getLogger(__name__)
            # 设置日志格式
            format_str = logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
            # 设置日志级别
            self.logger.setLevel(logging.INFO)
            # 往屏幕上输出
            sh = logging.StreamHandler()
            # 设置屏幕上显示的格式
            sh.setFormatter(format_str)
            self.logger.addHandler(sh)
        else:
            self.logger = logger
        self.num_sampled = num_sampled
        self.lr = lr
        self.epoch = epoch
        self.batch_size = 64
        if is_training:
            self.vocab, self.word2id = self.build_vocab()
            self.vocab_size = len(self.vocab)
            self.train(self.batch_size)
            self.save_model(self.w2v, self.sess, model_path)
        else:
            with tf.Graph().as_default():
                self.sess = tf.Session()
                with self.sess.as_default():
                    self.w2v = self.load_model(self.sess, model_path)

    def build_vocab(self):
        vocab = set()
        for sentence in self.sentences:
            vocab |= set(sentence)
        vocab = list(vocab)
        word2id = {word: i for i, word in enumerate(vocab)}
        self.logger.info('vocabulary build over. Found {0} words'.format(len(vocab)))
        return vocab, word2id

    def iter_batch(self, sentences, batch_size, shuffle=True):
        if shuffle:
            random.shuffle(sentences)
        batch_inputs, batch_labels = [], []
        for k, sentence in enumerate(sentences):
            for i in range(len(sentence)):
                start = max(0, i - self.window)
                end = min(len(sentence), i + self.window + 1)
                for idx in range(start, end):
                    if idx == i:
                        continue
                    else:
                        input_id = self.word2id.get(sentence[i])
                        label_id = self.word2id.get(sentence[idx])
                        if not (input_id and label_id):
                            continue
                        batch_inputs.append(input_id)
                        batch_labels.append(label_id)
            if len(batch_inputs) >= batch_size:
                yield batch_inputs, batch_labels

    def train(self, batch_size):
        with tf.Graph().as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                self.w2v = TFWord2Vec(self.vocab_size, self.size, self.num_sampled, self.lr)
                train_writer = tf.summary.FileWriter('./280', self.sess.graph)
                self.sess.run(tf.global_variables_initializer())
                for _epoch in range(self.epoch):
                    for i, (batch_inputs, batch_labels) in enumerate(self.iter_batch(self.sentences, batch_size)):
                        batch_inputs = np.array(batch_inputs, dtype=np.int32)
                        batch_labels = np.array(batch_labels, dtype=np.int32)
                        batch_labels = np.reshape(batch_labels, [len(batch_labels), 1])

                        feed_dict = {
                            self.w2v.train_inputs: batch_inputs,
                            self.w2v.train_labels: batch_labels
                        }
                        _, merge_summary, global_step, loss_val = self.sess.run(
                            [self.w2v.train_op, self.w2v.merge_summary, self.w2v.global_step, self.w2v.loss],
                            feed_dict=feed_dict
                        )
                        train_writer.add_summary(merge_summary, global_step)
                        print('Epoch: {0}/{1}, step: {2}, batch: {3}, loss: {4}'.format(
                            _epoch, self.epoch, global_step, i+1, loss_val))
                    self.get_words_vector(['萧炎', '灵魂'])
                    print(self.cal_similarity(['萧炎', '灵魂']))

    def save_model(self, model, sess, path):
        if os.path.isfile(path):
            raise RuntimeError('the save path should be a dir')
        if not os.path.isdir(path):
            os.makedirs(path)
        checkpoint_path = os.path.join(path, "tfw2v.ckpt")
        model.saver.save(sess, checkpoint_path)
        pickle.dump(self.word2id, open(os.path.join(path, "word2id.pkl"), 'wb'))
        self.logger.info("model saved")

    def load_model(self, session, path):
        self.word2id = pickle.load(open(os.path.join(path, "word2id.pkl"), 'rb'))
        self.vocab_size = len(self.word2id)
        model = TFWord2Vec(self.vocab_size, self.size, self.num_sampled, self.lr)
        checkpoint_path = os.path.join(path, "tfw2v.ckpt")
        if tf.train.checkpoint_exists(checkpoint_path):
            model.saver.restore(session, checkpoint_path)
        self.logger.info('model load successfully.')
        return model

    def get_words_vector(self, words):
        words_id = [self.word2id[word] for word in words]
        words_vector = self.sess.run(self.w2v.test_word_vector,
                                     feed_dict={self.w2v.test_word_id: words_id})
        return words_vector

    def cal_similarity(self, test_words, topk=10):
        test_word_id_list = [self.word2id[word] for word in test_words]
        sim_matrix = self.sess.run(self.w2v.similarity, feed_dict={self.w2v.test_word_id: test_word_id_list})
        sim_mean = np.mean(sim_matrix)
        sim_var = np.mean(np.square(sim_matrix-sim_mean))
        test_words = []
        near_words = []
        for i in range(test_word_id_list.__len__()):
            test_words.append(self.vocab[test_word_id_list[i]])
            nearst_id = (-sim_matrix[i, :]).argsort()[1: topk + 1]
            nearst_word = [self.vocab[x] for x in nearst_id]
            near_words.append(nearst_word)
        return test_words, near_words, sim_mean, sim_var
