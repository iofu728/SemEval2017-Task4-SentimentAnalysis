'''
@Author: gunjianpan
@Date:   2019-04-12 15:13:09
@Last Modified by:   gunjianpan
@Last Modified time: 2019-04-13 21:27:09
'''
import codecs
import html
import numpy as np
import os
import pickle
import random
import re
import subprocess
import tensorflow as tf
import time
import urllib3
import warnings

from constant import *
from bs4 import BeautifulSoup, NavigableString
from numba import jit
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")

start = []
sentences_map = {}

text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
               'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
              'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens

    # corpus from which the word statistics are going to be used
    # for word segmentation
    segmenter="twitter",

    # corpus from which the word statistics are going to be used
    # for spell correction
    corrector="twitter",

    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)


@jit
def fastF1(result, predict):
    ''' f1 score '''
    true_total, r_total, p_total, p, r = 0, 0, 0, 0, 0
    total_list = []
    for trueValue in range(num_class):
        trueNum, recallNum, precisionNum = 0, 0, 0
        for index, values in enumerate(result):
            if values == trueValue:
                recallNum += 1
                if values == predict[index]:
                    trueNum += 1
            if predict[index] == trueValue:
                precisionNum += 1
        R = trueNum / recallNum if recallNum else 0
        P = trueNum / precisionNum if precisionNum else 0
        true_total += trueNum
        r_total += recallNum
        p_total += precisionNum
        p += P
        r += R
        f1 = (2 * P * R) / (P + R) if (P + R) else 0
        print(id2label[trueValue], 'P: {:.2f}%, R: {:.2f}%, Macro_f1: {:.2f}%'.format(
            P * 100, R * 100, f1 * 100))
        total_list.append([P, R, f1])
    p /= num_class
    r /= num_class
    micro_r = true_total / r_total
    micro_p = true_total / p_total
    macro_f1 = (2 * p * r) / (p + r) if (p + r) else 0
    micro_f1 = (2 * micro_p * micro_r) / (micro_p +
                                          micro_r) if (micro_p + micro_r) else 0
    accuracy = true_total / len(result)
    print('P: {:.2f}%, R: {:.2f}%, Micro_f1: {:.2f}%, Macro_f1: {:.2f}%, Accuracy: {:.2f}'.format(
        p*100, r*100, micro_f1 * 100, macro_f1 * 100, accuracy * 100))
    return p, r, macro_f1, micro_f1, total_list, accuracy


def load_result_f1(result, predict):
    ''' load result calculate f1 score '''
    p, r, macro_f1, micro_f1, total_list, accuracy = fastF1(result, predict)
    result_file = [r, p, macro_f1, accuracy]
    result_file = [*result_file, *[ii[0] for ii in total_list]]
    result_file = [time_str(), *['%.2f' % (ii * 100) for ii in result_file]]
    with open('{}result.md'.format(result_dir), 'a') as f:
        f.write('|'.join(result_file) + '\n')
    return p, r, macro_f1, micro_f1, total_list, accuracy


def begin_time():
    """
    multi-version time manage
    """
    global start
    start.append(time.time())
    return len(start) - 1


def end_time(version):
    termSpend = time.time() - start[version]
    print(str(termSpend)[0:5])


def dump_bigger(data, output_file):
    """
    pickle.dump big file which size more than 4GB
    """
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(data, protocol=4)
    with open(output_file, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def load_bigger(input_file):
    """
    pickle.load big file which size more than 4GB
    """
    max_bytes = 2**31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(input_file)
    with open(input_file, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    return pickle.loads(bytes_in)


def load_embedding_from_txt(word2vec_model_name: str, index2word: dict, return_dict=False):
    import word2vec

    word2vec_model_path = '%s%s.pkl' % (pickle_dir, word2vec_model_name)
    if not os.path.exists(word2vec_model_path):
        word2vec_model_path = word2vec_model_path.replace('.pkl', '.txt')

    print("using pre-trained word embedding:", word2vec_model_path)
    if '.pkl' in word2vec_model_path:
        word2vec_dict = load_bigger(word2vec_model_path)
    else:
        embed = word2vec.load(word2vec_model_path, kind='txt')
        word2vec_dict = {w: v for w, v in zip(embed.vocab, embed.vectors)}
    if return_dict:
        return word2vec_dict

    vocab_size = len(index2word)
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)
    word_embedding_list = [word2vec_dict[jj] if jj in word2vec_dict else np.random.uniform(
        -bound, bound, embedding_dim) for ii, jj in index2word.items()]
    word_embedding_list[0] = np.zeros(embedding_dim)
    count_exist = len([1 for ww in index2word.values() if ww in word2vec_dict])
    count_not_exist = vocab_size - count_exist

    word_embedding_final = np.array(word_embedding_list)
    embed_dir = '%s%s_%d.pkl' % (pickle_dir, word2vec_model_name, vocab_size)

    dump_bigger([word_embedding_final, count_exist,
                 count_not_exist], embed_dir)
    return word_embedding_final, count_exist, count_not_exist


def load_embedding(sess, index2word: dict, W, word2vec_model_name: str):
    ''' load embedding '''
    embed_dir = '%s%s_%d.pkl' % (
        pickle_dir, word2vec_model_name, len(index2word))
    if os.path.exists(embed_dir):
        word_embedding_final, count_exist, count_not_exist = load_bigger(
            embed_dir)
    else:
        word_embedding_final, count_exist, count_not_exist = load_embedding_from_txt(
            word2vec_model_name, index2word)

    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)
    t_assign_embedding = tf.assign(W, word_embedding)
    sess.run(t_assign_embedding)
    print("word. exists embedding:{} ;word not exist embedding: {}".format(
        count_exist, count_not_exist))
    print("using pre-trained word embedding.ended...")


def get_embeddings(corpus, dim):
    vectors = load_embedding_from_txt(corpus, {}, True)
    vocab_size = len(vectors)
    print('Loaded %s word vectors.' % vocab_size)
    wv_map = {}
    pos = 0
    # +1 for zero padding token and +1 for unk
    emb_matrix = np.ndarray((vocab_size + 2, dim), dtype='float32')
    for i, (word, vector) in enumerate(vectors.items()):
        if len(vector) > 199:
            pos = i + 1
            wv_map[word] = pos
            emb_matrix[pos] = vector

    pos += 1
    wv_map["<unk>"] = pos
    emb_matrix[pos] = np.random.uniform(low=-0.05, high=0.05, size=dim)

    return emb_matrix, wv_map


def pad_middle(sent: str, max_len: int, types: int = 1, pad_type: int = 0):
    ''' add padding elements (i.e. dummy word tokens) to fill the sentence to max_len '''
    sent = sent.split()
    entity1 = [sent[0]]
    middle_word = sent[1:-1]
    entity2 = [sent[-1]]
    num_pads = max_len - len(entity1) - len(entity2) - len(middle_word)
    if not pad_type:  # text_cnn model
        padding = num_pads * ['<pad>']
    else:  # bert model
        padding = num_pads * ['[PAD]']

    if not types:  # [PAD] site
        return ' '.join([*padding, *entity1, *middle_word, *entity2])
    elif types == 1:
        return ' '.join([*entity1, *padding, *middle_word, *entity2])
    elif types == 2:
        return ' '.join([*entity1, *middle_word, *padding, *entity2])
    else:
        return ' '.join([*entity1, *middle_word, *entity2, *padding])


def load_data():
    ''' load data '''
    train_data = sum([load_a_file(ii) for ii in train_data_list], [])
    test_data = load_a_file(test_data_path)
    return train_data, test_data


def load_a_file(file_name):
    ''' load a file '''
    global sentences_map
    temp_data_list = []
    with codecs.open(file_name, 'r', encoding='utf-8') as f:
        temp_data = [jj.strip() for jj in f.readlines()]
    label_error = 0
    sentence_error = 0
    for ii in temp_data:
        try:
            tid, label, sentence = ii.split(None, 2)
        except:
            print(ii)
            continue
        if sentence[0] == '"':
            sentence = sentence[1:]
        if sentence[-1] == '"':
            sentence = sentence[:-1]
        sentence = sentence.replace('\\u2019', u'\'').replace(
            '\\u002c', ',').replace('""""', '"').replace('""', '"')
        if 'http://t.co' in sentence:
            sentence = sentence.split('http://t.co', 1)[0]
        sentence = str(sentence)

        if tid in sentences_map:
            if sentences_map[tid][0] != sentence:
                print(tid, str(sentences_map[tid][0]), '\n' + str(sentence))
                sentence_error += 1
            elif sentences_map[tid][1] != label:
                label_error += 1
        else:
            sentences_map[tid] = [sentence, label]
            temp_data_list.append([sentence, label2id[label]])
    print('label error', label_error, 'sentence error', sentence_error)
    return temp_data_list


def clean_text(text):
    text = text.rstrip()

    if '""' in text:
        if text[0] == text[-1] == '"':
            text = text[1:-1]
        text = text.replace('\\""', '"')
        text = text.replace('""', '"')

    text = text.replace('\\""', '"')

    text = html.unescape(text)
    text = ' '.join(text.split())
    return text


def time_str(timestamp: int = -1, format: str = '%Y-%m-%d %H:%M:%S'):
    if timestamp > 0:
        return time.strftime(format, time.localtime(timestamp))
    return time.strftime(format, time.localtime(time.time()))
