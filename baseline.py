'''
@Author: gunjianpan
@Date:   2019-04-13 14:54:51
@Last Modified by:   gunjianpan
@Last Modified time: 2019-04-13 17:19:34
'''
import numpy

from constant import label2id, embedding_dim
from data_load import SemEvalDataLoader
from model_sklearn import nbow_model, bow_model, eval_clf
from util import load_result_f1, get_embeddings

numpy.random.seed(1337)  # for reproducibility


def nbow():
    ''' NBOW baseline '''
    WV_CORPUS = "origin"

    embeddings, word_indices = get_embeddings(
        corpus=WV_CORPUS, dim=embedding_dim)

    train_set = SemEvalDataLoader(verbose=False, ekphrasis=True).get_data(task="A",
                                                                          years=None,
                                                                          datasets=None,
                                                                          only_semEval=True)
    test_data = SemEvalDataLoader(
        verbose=False, ekphrasis=True).get_gold(task="A")
    X = [obs[1] for obs in train_set]
    y = [label2id[obs[0]] for obs in train_set]

    X_test = [obs[1] for obs in test_data]
    y_test = [label2id[obs[0]] for obs in test_data]

    task = 'clf'
    print("-----------------------------")
    if task == 'clf':
        print('LogisticRegression')
    else:
        print("LinearSVC")

    bow = bow_model(task)
    bow.fit(X, y)
    predict = bow.predict(X_test)
    results = eval_clf(predict, y_test)
    for res, val in results.items():
        print("{}: {:.3f}".format(res, val))
    load_result_f1(predict, y_test)

    nbow = nbow_model(task, embeddings, word_indices)
    nbow.fit(X, y)
    predict = nbow.predict(X_test)
    results = eval_clf(predict, y_test)
    for res, val in results.items():
        print("{}: {:.3f}".format(res, val))
    load_result_f1(predict, y_test)
    print("-----------------------------")


if __name__ == '__main__':
    nbow()
