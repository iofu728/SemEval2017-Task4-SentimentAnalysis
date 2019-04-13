'''
@Author: gunjianpan
@Date:   2019-04-13 16:16:52
@Last Modified by:   gunjianpan
@Last Modified time: 2019-04-13 17:23:58
'''

import nltk
import numpy
import os
import pickle

from cachetools import cached
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_similarity_score, f1_score, \
    precision_score, recall_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from util import text_processor


def eval_clf(y_test, y_p):
    results = {
        "f1": f1_score(y_test, y_p, average='macro'),
        "recall": recall_score(y_test, y_p, average='macro'),
        "precision": precision_score(y_test, y_p, average='macro'),
        "accuracy": accuracy_score(y_test, y_p)
    }

    return results


def bow_model(task, max_features=10000):
    if task == "clf":
        algo = LogisticRegression(C=0.6, random_state=0,
                                  class_weight='balanced')
    elif task == "reg":
        algo = SVR(kernel='linear', C=0.6)
    else:
        raise ValueError("invalid task!")

    word_features = TfidfVectorizer(ngram_range=(1, 1),
                                    tokenizer=lambda x: x,
                                    analyzer='word',
                                    min_df=5,
                                    # max_df=0.9,
                                    lowercase=False,
                                    use_idf=True,
                                    smooth_idf=True,
                                    max_features=max_features,
                                    sublinear_tf=True)

    model = Pipeline([
        # ('preprocess', CustomPreProcessor(text_processor, to_list=True)),
        ('bow-feats', word_features),
        ('normalizer', Normalizer(norm='l2')),
        ('clf', algo)
    ])

    return model


def nbow_model(task, embeddings, word2idx):
    if task == "clf":
        algo = LogisticRegression(C=0.6, random_state=0,
                                  class_weight='balanced')
    elif task == "reg":
        algo = SVR(kernel='linear', C=0.6)
    elif task == 'dt':
        algo = DecisionTreeClassifier()
    else:
        raise ValueError("invalid task!")

    embeddings_features = NBOWVectorizer(aggregation=["mean"],
                                         embeddings=embeddings,
                                         word2idx=word2idx,
                                         stopwords=False)

    model = Pipeline([
        # ('preprocess', CustomPreProcessor(text_processor, to_list=True)),
        ('embeddings-feats', embeddings_features),
        ('normalizer', Normalizer(norm='l2')),
        ('clf', algo)
    ])

    return model


class CustomPreProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, pp, to_list=False):
        self.pp = pp
        self.to_list = to_list

    @cached(cache={})
    def pre_process_doc(self, doc):
        if isinstance(doc, tuple) or isinstance(doc, list):
            return [self.pp.pre_process_doc(d) for d in doc]
        else:
            return self.pp.pre_process_doc(doc)

    def pre_process_steps(self, X):
        for x in tqdm(X, desc="PreProcessing..."):
            yield self.pre_process_doc(x)

    def transform(self, X, y=None):
        if self.to_list:

            if os.path.exists('{}.pickle'.format(len(X))):
                with open('{}.pickle'.format(len(X)), 'rb') as handle:
                    processed = pickle.load(handle)
            else:
                processed = list(self.pre_process_steps(X))
                with open('{}.pickle'.format(len(X)), 'wb') as handle:
                    pickle.dump(processed, handle)
            return numpy.array(processed)
        else:
            processed = self.pre_process_steps(X)
            return processed

    def fit(self, X, y=None):
        return self


class NBOWVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, aggregation, embeddings=None, word2idx=None,
                 stopwords=True):
        self.aggregation = aggregation
        self.embeddings = embeddings
        self.word2idx = word2idx
        self.dim = embeddings[0].size
        self.stopwords = stopwords
        self.stops = set(nltk.corpus.stopwords.words('english'))

    def aggregate_vecs(self, vectors):
        feats = []
        for method in self.aggregation:
            if method == "sum":
                feats.append(numpy.sum(vectors, axis=0))
            if method == "mean":
                feats.append(numpy.mean(vectors, axis=0))
            if method == "min":
                feats.append(numpy.amin(vectors, axis=0))
            if method == "max":
                feats.append(numpy.amax(vectors, axis=0))
        return numpy.hstack(feats)

    def transform(self, X, y=None):
        docs = []
        for doc in X:
            vectors = []
            for word in doc:
                if word not in self.word2idx:
                    continue
                if not self.stopwords and word in self.stops:
                    continue
                vectors.append(self.embeddings[self.word2idx[word]])
            if len(vectors) == 0:
                vectors.append(numpy.zeros(self.dim))
            feats = self.aggregate_vecs(numpy.array(vectors))
            docs.append(feats)
        return docs

    def fit(self, X, y=None):
        return self
