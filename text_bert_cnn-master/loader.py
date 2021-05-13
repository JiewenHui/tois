import collections
import codecs
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from bert import tokenization
from text_model import TextConfig
# -*- coding:utf-8-*-
import numpy as np
import random,os,math
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.keyedvectors import KeyedVectors
import sklearn
import multiprocessing
import time
import pickle 
from collections import defaultdict
import evaluation
import string
from nltk import stem
from tqdm import tqdm
import chardet
import re
import config
from functools import wraps
import nltk
from nltk.corpus import stopwords
from numpy.random import seed
import math



class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, label=None):
        """
        构造bert模型样本的类
        Args:
          guid: 样本的编码，表示第几条数据，不是模型要输入的对应参数；
          text_a: 第一个序列文本，对应我们数据集要分类的文本；
          text_b: 第二个序列文本，是bert模型在sequence pair 任务要输入的文本，在我们这个场景不需要，设置为None;
          label: 文本标签
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class TextProcessor(object):
    """按照InputExample类形式载入对应的数据集"""

    """load train examples"""
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "train_all.txt")), "train")

    """load dev examples"""
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "devvv.txt")), "dev")

    """load test examples"""
    def get_test_examples(self, data_dir):
          return self._create_examples(
              self._read_file(os.path.join(data_dir, "testtt.txt")), "test")

    """set labels"""
    def get_labels(self):
        return ['0','1']

    """read file"""
    def _read_file(self, input_file):
        with codecs.open(input_file, "r",encoding='utf-8') as f:
            lines = []
            for line in f.readlines():
                try:
                    line=line.strip().split('\t')
                    assert len(line)==3
                    lines.append(line)
                except:
                    pass
            np.random.shuffle(lines)
            return lines

    """create examples for the data set """
    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
          guid = "%s-%s" % (set_type, i)
          text_a = tokenization.convert_to_unicode(line[0])
          text_b = tokenization.convert_to_unicode(line[1])
          label = tokenization.convert_to_unicode(line[2])
          examples.append(
              InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

def convert_examples_to_features(examples,label_list, max_seq_length,tokenizer,name):
    """
    将所有的InputExamples样本数据转化成模型要输入的token形式，最后输出bert模型需要的四个变量；
    input_ids：就是text_a(分类文本)在词库对应的token，按字符级；
    input_mask：bert模型mask训练的标记，都为1；
    segment_ids：句子标记，此场景只有text_a,都为0；
    label_ids：文本标签对应的token，不是one_hot的形式；
    """
    col_name = ["input_ids","input_mask","segment_ids","label_ids","question_length","answer_length","question","answer"]
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    input_data=[]
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = tokenizer.tokenize(example.text_b)
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        if tokens_b:
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        question_length,answer_length = [],[]
        pairs = []
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        question_length.append(len(tokens_a)+2)
        segment_ids.append(0)
        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            answer_length.append(len(tokens_b)+1)
            segment_ids.append(1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        
        if ex_index < 3:
            tf.logging.info("*** Example ***")
            tf.logging.info("guid: %s" % (example.guid))
            tf.logging.info("question_length: %s" % (question_length))
            tf.logging.info("answer_length: %s" % (answer_length))
            tf.logging.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            tf.logging.info("label: {} (id = {})".format(example.label, label_id))

        features = collections.OrderedDict()
        features["input_ids"] = input_ids
        features["input_mask"] = input_mask
        features["segment_ids"] = segment_ids
        features["label_ids"] = label_id
        features["question_length"] = question_length
        features["answer_length"] = answer_length
        features["question"] = example.text_a
        features["answer"] = example.text_b
        input_data.append(features)

    if not os.path.exists('/root/hwj/new_data/ms_small/'+name+'.csv'):
        print('heihiehiehiehie')
        df = pd.DataFrame.from_dict(input_data)
        df.to_csv('/root/hwj/new_data/ms_small/'+name+'.csv',encoding='utf-8',index=False)
    return input_data

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def batch_iter(input_data,batch_size,haha=True, df = None):
    """
    将样本的四个tokens形式的变量批量的输入给模型；
    """
    if haha:
        batch_ids,batch_mask,batch_segment,batch_label=[],[],[],[]
        #========模型具体需要的变量=========#
        question_length,answer_length = [],[]
        #=================================#

        for features in input_data:
            if len(batch_ids) == batch_size:
                yield batch_ids,batch_mask,batch_segment,batch_label,question_length,answer_length
                batch_ids, batch_mask, batch_segment, batch_label = [], [], [], []
                question_length,answer_length = [],[]

            batch_ids.append(features['input_ids'])
            batch_mask.append(features['input_mask'])
            batch_segment.append(features['segment_ids'])
            batch_label.append(features['label_ids'])

            question_length.append(features['question_length'])
            answer_length.append(features['answer_length'])

        if len(batch_ids) != 0:
            yield batch_ids, batch_mask, batch_segment, batch_label,question_length,answer_length
    else:
        pairs = []


        for index, row in df.iterrows():

            batch_ids = row['input_ids']
            batch_ids = [int(n) for n in batch_ids[1:-1].split(',')]
            batch_mask = row['input_mask']
            batch_mask = [int(n) for n in batch_mask[1:-1].split(',')]
            batch_segment = row['segment_ids']
            batch_segment = [int(n) for n in batch_segment[1:-1].split(',')]
            batch_label = row['label_ids']
            question_length = row['question_length']
            question_length = [int(n) for n in question_length[1:-1].split(',')]
            answer_length = row['answer_length']
            answer_length = [int(n) for n in answer_length[1:-1].split(',')]

            pairs.append((batch_ids,batch_mask,batch_segment,batch_label,question_length,answer_length))
            input_num = 6

        n_batches = int(len(pairs) * 1.0 / batch_size)

        for i in range(0, n_batches):
            batch = pairs[i * batch_size:(i + 1) * batch_size]
            yield [np.array([pair[i] for pair in batch]) for i in range(input_num)]
        batch = pairs[n_batches * batch_size:] + [pairs[n_batches *
                                                    batch_size]] * (batch_size - len(pairs) + n_batches * batch_size)
        yield [np.array([pair[i] for pair in batch]) for i in range(input_num)]

def load(dataset, filter=False):
    data_dir = dataset
    datas = []
    for data_name in ['train.csv','dev.csv']:
        data_file = os.path.join(data_dir, data_name)
        data = pd.read_csv(data_file).fillna('N')
        if filter == True:
            datas.append(removeUnanswerdQuestion(data))
        else:
            datas.append(data)
    return tuple(datas)

def removeUnanswerdQuestion(df):
    counter = df.groupby("question").apply(lambda group: sum(group["label_ids"]))
    questions_have_correct = counter[counter > 0].index
    counter = df.groupby("question").apply(
        lambda group: sum(group["label_ids"] == 0))
    questions_have_uncorrect = counter[counter > 0].index
    counter = df.groupby("question").apply(lambda group: len(group["label_ids"]))
    questions_multi = counter[counter > 1].index

    return df[df["question"].isin(questions_have_correct) & df["question"].isin(questions_have_correct) & df["question"].isin(questions_have_uncorrect)].reset_index()
