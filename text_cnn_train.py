'''
@Author: gunjianpan
@Date:   2019-04-12 22:12:46
@Last Modified by:   gunjianpan
@Last Modified time: 2019-04-14 09:45:50
'''

import tensorflow as tf
import numpy as np
from text_cnn_big import TextCNN
import pickle
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from constant import *
from util import load_embedding, pad_middle, load_result_f1, time_str
from data_load import SemEvalDataLoader

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float("learning_rate", 0.0003, "learning rate")
tf.app.flags.DEFINE_integer(
    "batch_size", 320, "Batch size for training/evaluating.")  # 批处理的大小 32-->128
tf.app.flags.DEFINE_integer(
    "decay_steps", 1000, "how many steps before decay learning rate.")  # 6000批处理的大小 32-->128
tf.app.flags.DEFINE_float(
    "decay_rate", 1.0, "Rate of decay for learning rate.")  # 0.65一次衰减多少
tf.app.flags.DEFINE_string("ckpt_dir", "checkpoint/",
                           "checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_len", 200, "max sentence length")
tf.app.flags.DEFINE_integer("embed_size", 300, "embedding size")
tf.app.flags.DEFINE_boolean(
    "is_training_flag", True, "is training.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs", 320, "number of epochs to run.")
tf.app.flags.DEFINE_integer(
    "validate_every", 10, "Validate every validate_every epochs.")  # 每10轮做一次验证
tf.app.flags.DEFINE_boolean("use_embedding", True,
                            "whether to use embedding or not.")
tf.app.flags.DEFINE_integer(
    "num_filters", 256, "number of filters")  # 256--->512
tf.app.flags.DEFINE_string("embedding_name", "origin",
                           "word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_string("name_scope", "cnn", "name scope value.")
tf.app.flags.DEFINE_boolean(
    "multi_label_flag", True, "use multi label or single label.")
tf.app.flags.DEFINE_integer("pad_type", 0, "pad type")
tf.app.flags.DEFINE_boolean("ekphrasis", True, "ekphrasis type")

filter_sizes = [6, 7, 8]


def main(_):
    ''' 1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction) '''

    word2index, label2index, trainX, trainY, vaildX, vaildY, testX, testY = load_data()
    max_f1, max_p, max_r, max_acc, test_f1, test_acc, test_p, test_r = [0] * 8
    voc_size, num_class, max_time_str = [len(word2index), len(label2index), '']

    print("cnn_model.voc_size: {}, num_class: {}".format(voc_size, num_class))

    num_examples, FLAGS.sentence_len = trainX.shape
    print("num_examples of training:", num_examples,
          ";sentence_len:", FLAGS.sentence_len)

    ''' print some message for debug purpose '''
    print("trainX[0:10]:", trainX[0:10])
    print("trainY[0]:", trainY[0:10])
    print("train_y_short:", trainY[0])

    ''' 2.create session '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        ''' Instantiate Model '''
        textCNN = TextCNN(filter_sizes, FLAGS.num_filters, num_class, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,
                          FLAGS.decay_rate, FLAGS.sentence_len, voc_size, FLAGS.embed_size, multi_label_flag=FLAGS.multi_label_flag)
        ''' Initialize Save '''
        saver = tf.train.Saver()
        if False:
            # if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint.")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding:
                index2word = {v: k for k, v in word2index.items()}
                load_embedding(sess, index2word,
                               textCNN.Embedding, FLAGS.embedding_name)
        current_epoch = sess.run(textCNN.epoch_step)

        ''' 3.feed data & training '''
        number_of_training_data = len(trainX)
        batch_size = FLAGS.batch_size
        iteration = 0
        for epoch in range(current_epoch, FLAGS.num_epochs):
            loss, counter = 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size), range(batch_size, number_of_training_data, batch_size)):
                iteration = iteration+1
                if not epoch and not counter:
                    print("trainX[start:end]:", trainX[start:end])
                feed_dict = {
                    textCNN.input_x: trainX[start:end], textCNN.dropout_keep_prob: 0.8, textCNN.is_training_flag: FLAGS.is_training_flag}
                if not FLAGS.multi_label_flag:
                    feed_dict[textCNN.input_y] = trainY[start:end]
                else:
                    feed_dict[textCNN.input_y_multilabel] = trainY[start:end]
                curr_loss, lr, _ = sess.run(
                    [textCNN.loss_val, textCNN.learning_rate, textCNN.train_op], feed_dict)
                loss, counter = loss + curr_loss, counter + 1
                if not counter % 50:
                    print("%s Epoch %d\tBatch %d\tTrain Loss:%.3f\tLearning rate:%.5f" % (
                        time_str(), epoch, counter, loss/float(counter), lr))

            ''' vaild model '''
            if not epoch % FLAGS.validate_every:
                eval_loss, f1, r, p, acc = do_eval(
                    sess, textCNN, vaildX, vaildY, num_class)
                print("Epoch %d Validation Loss:%.3f\tR:%.3f\tP:%.3f\tF1 Score:%.3f\tacc:%.3f" % (
                    epoch, eval_loss, r*100, p*100, f1*100, acc*100))
                if r > max_r:
                    max_time_str = time_str()
                    max_f1, max_acc, max_p, max_r = [f1, acc, p, r]
                    eval_loss, test_f1, test_r, test_p, test_acc = do_eval(
                        sess, textCNN, testX, testY, num_class)
                    print("Test Loss:%.3f|%.3f|%.3f|%.3f|%.3f" %
                          (eval_loss, test_r*100, test_p*100, test_f1*100, test_acc*100))
                ''' save model to checkpoint '''
                save_path = FLAGS.ckpt_dir + "model.ckpt"
                saver.save(sess, save_path, global_step=epoch)
            sess.run(textCNN.epoch_increment)

            ''' test model '''
            if not epoch % 100:
                eval_loss, f1, r, p, acc = do_eval(
                    sess, textCNN, testX, testY, num_class)
                print("%s Epoch %d Test Loss:%.3f\tR:%.3f\tP:%.3f\tF1 Score:%.3f\tacc:%.3f" % (
                    time_str(), epoch, eval_loss, r*100, p*100, f1*100, acc*100))

        ''' print train best '''
        print("%s Train MAX F1_micro:%.3f|%.3f|%.3f|%.3f" %
              (max_time_str, max_r * 100, max_p * 100, max_f1 * 100, max_acc * 100))
        print("%s Test F1_micro:%.3f|%.3f|%.3f|%.3f" %
              (max_time_str, test_r * 100, test_p * 100, test_f1 * 100, test_acc * 100))


def do_eval(sess, textCNN, evalX, evalY, num_class):
    evalX = evalX[0:3000]
    evalY = evalY[0:3000]
    number_examples = len(evalX)
    eval_loss, eval_counter, eval_f1_score, eval_p, eval_r = 0.0, 0, 0.0, 0.0, 0.0
    batch_size = 1
    predict = []

    for start, end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples + batch_size, batch_size)):
        ''' evaluation in one batch '''
        feed_dict = {textCNN.input_x: evalX[start:end], textCNN.input_y_multilabel: evalY[start:end], textCNN.dropout_keep_prob: 1.0,
                     textCNN.is_training_flag: False}
        current_eval_loss, logits = sess.run(
            [textCNN.loss_val, textCNN.logits], feed_dict)
        predict = [*predict, np.argmax(np.array(logits[0]))]
        eval_loss += current_eval_loss
        eval_counter += 1
    evalY = [np.argmax(ii) for ii in evalY]
    print(evalY[:10], predict[:10])

    if not FLAGS.multi_label_flag:
        predict = [int(ii > 0.5) for ii in predict]
    p, r, f1, _, _, acc = load_result_f1(predict, evalY)
    return eval_loss/float(eval_counter), f1, r, p, acc


def load_data():
    data_path = '{}data_{}.pkl'.format(pickle_dir, FLAGS.ekphrasis)
    if os.path.exists(data_path):
        print(11111)
        train_set, test_data = pickle.load(open(data_path, 'rb'))
    else:
        train_set = SemEvalDataLoader(verbose=False, ekphrasis=FLAGS.ekphrasis).get_data(task="A",
                                                                                         years=None,
                                                                                         datasets=None,
                                                                                         only_semEval=True)
        test_data = SemEvalDataLoader(
            verbose=False, ekphrasis=FLAGS.ekphrasis).get_gold(task="A")
        pickle.dump([train_set, test_data], open(data_path, 'wb'))
    X = [obs[1] for obs in train_set]
    y = [label2id[obs[0]] for obs in train_set]
    X_test = [obs[1] for obs in test_data]
    y_test = [label2id[obs[0]] for obs in test_data]

    sentences_len = [len(ii.split()) for ii in [*X, *X_test]]
    sent_size = max(sentences_len)
    pad_type = FLAGS.pad_type

    sent_re = [pad_middle(ii, sent_size, pad_type) for ii in X]
    test_sent_out = [pad_middle(ii, sent_size, pad_type) for ii in X_test]
    wordlist = ' '.join([*X, *X_test]).split()
    wordlist = sorted(list(set(wordlist)))
    wordlist = ['<pad>', *wordlist]
    word2index = {w: i for i, w in enumerate(wordlist)}
    index2word = {i: w for w, i in word2index.items()}
    vocab_size = len(word2index)

    input_x = [np.asarray([word2index[word]
                           for word in ii.split()]) for ii in sent_re]
    output = [np.eye(num_class)[label] for label in y]

    test_X = [np.asarray([word2index[word] for word in ii.split()])
              for ii in test_sent_out]
    test_Y = [np.eye(num_class)[ii] for ii in y_test]

    label2index = {ii: ii for ii in range(num_class)}
    train_X, X_test, train_Y, y_test = train_test_split(
        input_x, output, test_size=0.25)
    train_X = pd.DataFrame(train_X)
    print('Train', len(train_X), 'Test', len(X_test), 'Valid', len(test_X))

    return word2index, label2index, train_X, train_Y, X_test, y_test, test_X, test_Y


if __name__ == "__main__":
    tf.app.run()
