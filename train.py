# -*- coding: utf-8 -*-
# @Time        : 2020/5/24 21:26
# @Author      : tianyunzqs
# @Description : 

import os
import sys
import re
import jieba
project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_path)
from word2vec_model import Word2Vec


def load_stopwords(path):
    stopwords = set([line.strip() for line in open(path, "r", encoding="utf-8").readlines() if line.strip()])
    return stopwords


def remove_brackets(text):
    return re.sub(r'[(（][^(（]*?[)）]', '', text)


def text_segment(text, stopwords=None):
    return [word.strip() for word in jieba.lcut(text) if stopwords and word.strip() and word not in stopwords]


def load_train_data(stopwords):
    sentence_list = []
    with open('280.txt', encoding='gbk') as f:
        line = f.readline()
        while line:
            while '\n' in line:
                line = line.replace('\n', '')
            while ' ' in line:
                line = line.replace(' ', '')
            if len(line) > 0:  # 如果句子非空
                raw_words = list(jieba.cut(line, cut_all=False))
                dealed_words = []
                for word in raw_words:
                    if word not in stopwords and word not in ['qingkan520', 'www', 'com', 'http']:
                        dealed_words.append(word)
                sentence_list.append(dealed_words)
                if len(sentence_list) >= 100:
                    break
            line = f.readline()
    return sentence_list


if __name__ == '__main__':
    stop_words = load_stopwords('stop_words.txt')
    sents = load_train_data(stop_words)
    www = Word2Vec(model_path='./280_model', is_training=False)
    print(www.get_words_vector(['萧炎', '灵魂']))
