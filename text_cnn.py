'''
@Author: gunjianpan
@Date:   2019-04-12 20:56:54
@Last Modified by:   gunjianpan
@Last Modified time: 2019-04-13 14:22:37
'''

from constant import *
from data_load import SemEvalDataLoader
import pickle
import numpy as np
import tensorflow as tf
from util import load_embedding, pad_middle, fastF1

tf.reset_default_graph()

initializer = tf.random_normal_initializer(stddev=0.1)


def main():
    train_set = SemEvalDataLoader(verbose=False).get_data(task="A",
                                                          years=None,
                                                          datasets=None,
                                                          only_semeval=True)
    test_data = SemEvalDataLoader(verbose=False).get_gold(task="A")
    X = [obs[1] for obs in train_set]
    y = [label2id[obs[0]] for obs in train_set]
    X_test = [obs[1] for obs in test_data]
    y_test = [label2id[obs[0]] for obs in test_data]

    sentences_len = [len(ii.split()) for ii in [*X, *X_test]]
    sent_size = max(sentences_len)

    sent_re = [pad_middle(ii, sent_size) for ii in X]
    test_sent_out = [pad_middle(ii, sent_size) for ii in X_test]

    wordlist = ' '.join([*X, *X_test]).split()
    wordlist = sorted(list(set(wordlist)))
    wordlist = ['<pad>', *wordlist]
    word2index = {w: i for i, w in enumerate(wordlist)}
    index2word = {i: w for w, i in word2index.items()}
    vocab_size = len(word2index)

    print(vocab_size, sent_size)

    input_x = [np.asarray([word2index[word]
                           for word in ii.split()]) for ii in sent_re]
    output = [np.eye(num_class)[label] for label in y]

    X = tf.placeholder(tf.int32, [None, sent_size])
    Y = tf.placeholder(tf.int32, [None, num_class])  # batch_size num_class

    Embedding = tf.Variable(tf.random_uniform(
        [vocab_size, embedding_dim], -1.0, 1.0))

    embed = tf.nn.embedding_lookup(Embedding, X)
    embed = tf.expand_dims(embed, -1)

    pool_output = []
    for i, filter_size in enumerate(filter_sizes):
        # embed batch_size sentences_size embedding_dim -1
        filter_shape = [filter_size, embedding_dim, 1, num_filter]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
        b = tf.constant(0.1, shape=[num_filter])

        conv = tf.nn.conv2d(embed,  # 卷积
                            W,
                            strides=[1, 1, 1, 1],
                            padding='VALID')
        h = tf.nn.relu(tf.nn.bias_add(conv, b))  # 非线性激活
        # max_pooling
        pool = tf.nn.max_pool(h,
                              # [batch_size, filter_height, filter_width, channel]
                              ksize=[1, sent_size-filter_size+1, 1, 1],
                              strides=[1, 1, 1, 1],
                              padding='VALID')
        pool_output.append(pool)

    filter_total = len(filter_sizes)*num_filter  # 卷积核个数*通道数
    # h_pool : [batch_size(=6), output_height(=1), output_width(=1), channel(=1) * 3]
    h_pool = tf.concat(pool_output, num_filter)

    h_pool = tf.reshape(h_pool, shape=[-1, filter_total])

    Weights = tf.get_variable('W', shape=[
        filter_total, num_class], initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.Variable(tf.constant(0.1, shape=[num_class]))
    model = tf.nn.xw_plus_b(h_pool, Weights, bias)

    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))
    optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

    prediction = tf.nn.softmax(model)
    prediction = tf.argmax(prediction, 1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        load_embedding(sess, index2word, Embedding, 'datastories.twitter.300d')
        for i in range(5000):
            _, loss = sess.run([optimizer, cost], feed_dict={
                X: input_x, Y: output})
            if (i+1) % 1000 == 0:
                print('epoch:%d cost:%.6f' % ((i+1), loss))

        test_num = [[word2index[jj] for jj in ii.split()]
                    for ii in test_sent_out]
        predict = sess.run([prediction], feed_dict={X: test_num})[0]
        p, r, f1, _, _, acc = fastF1(predict[0], y_test)
        print("Train F1_micro:%.3f|%.3f|%.3f|%.3f" %
              (r * 100, p * 100, f1 * 100, acc * 100))
        pickle.dump(predict, open('{}predict.pkl'.format(pickle_dir), 'wb'))


if __name__ == '__main__':
    main()
