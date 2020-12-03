from col774_yelp_data.utils import json_reader
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from col774_yelp_data.utils import json_reader
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
import numpy as np
from tqdm import tqdm
from time import time
from sklearn.model_selection import train_test_split

print("Loading train and test data.")
train_corpus = []
train_y = []
for line in tqdm(json_reader("col774_yelp_data/train.json")):
    train_corpus.append(line['text'])
    train_y.append(line['stars'])
train_y = np.array(train_y).astype(np.uint8)

test_corpus = []
test_y = []
for line in tqdm(json_reader("col774_yelp_data/test.json")):
    test_corpus.append(line['text'])
    test_y.append(line['stars'])
test_y = np.array(test_y)

def _naive_bayes_scikit():
    start = time()
    vectorizer1 = CountVectorizer(stop_words='english', analyzer='word')
    X = vectorizer1.fit_transform(train_corpus)
    vocab = vectorizer1.vocabulary_

    vectorizer2 = CountVectorizer(vocabulary=vocab)
    test_X = vectorizer2.fit_transform(test_corpus)

    clf = MultinomialNB(alpha=1.0)
    clf.fit(X, train_y)

    print("time taken to train naive bayes on Scikit : {:.3f} seconds.".format(time()-start))

    preds = clf.predict(test_X)

    mask = preds == test_y
    return sum(mask) / len(mask)

def _linear_svm_scikit():
    start = time()
    vectorizer1 = CountVectorizer(stop_words='english', analyzer='word')
    X = vectorizer1.fit_transform(train_corpus) 
    vocab = vectorizer1.vocabulary_

    vectorizer2 = CountVectorizer(vocabulary=vocab)
    test_X = vectorizer2.fit_transform(test_corpus)

    svm = LinearSVC(C=1.0, tol=1e-5)
    svm.fit(X, train_y)

    print("time taken to train LinearSVC (SVM) on Scikit : {:.3f} seconds.".format(time()-start))

    preds = svm.predict(test_X)

    mask = preds == test_y
    return sum(mask) / len(mask)

def _linear_sgd_scikit():
    start = time()
    vectorizer1 = CountVectorizer(stop_words='english', analyzer='word')
    X = vectorizer1.fit_transform(train_corpus)
    vocab = vectorizer1.vocabulary_

    vectorizer2 = CountVectorizer(vocabulary=vocab)
    test_X = vectorizer2.fit_transform(test_corpus)

    clf = SGDClassifier(max_iter=1000, tol=1e-3)
    clf.fit(X, train_y)

    print("time taken to train gradient descent classifier on Scikit : {:.3f} seconds.".format(time()-start))

    preds = clf.predict(test_X)

    mask = preds == test_y
    return sum(mask) / len(mask)

def _tune_hyperparams():
    print("Tuning hyperparams")
    start = time()
    vectorizer1 = TfidfVectorizer(stop_words='english', analyzer='word')
    X = vectorizer1.fit_transform(train_corpus)

    X_train, X_test, y_train, y_test = train_test_split(X, train_y, test_size=0.20, shuffle=True)

    # c_values = [1e-5, 1e-3, 1, 5, 10]
    c_values = [0.1, 0.5, 2.5]
    accu = []

    for c in tqdm(c_values):
        svm = LinearSVC(C=c, tol=1e-5)
        svm.fit(X_train, y_train)

        preds = svm.predict(X_test)

        mask = preds == y_test
        acc = sum(mask) / len(mask)
        accu.append(acc)

    print(accu)

if __name__ == "__main__":

    # score_naive = _naive_bayes_scikit()
    # print("Accuracy for naive bayes is : {:.3f}".format(score_naive * 100))

    score_liblinear = _linear_svm_scikit()
    print("Accuracy for liblinear SVM is : {:.3f}".format(score_liblinear * 100))

    # score_clf = _linear_sgd_scikit()
    # print("Accuracy for SGD classifier is : {:.3f}".format(score_clf * 100))

    # _tune_hyperparams()