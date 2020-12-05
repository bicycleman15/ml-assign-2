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



def _load_data(train_path, test_path):
    train_corpus = []
    train_y = []
    test_corpus = []
    test_y = []
    for line in tqdm(json_reader(train_path)):
        train_corpus.append(line['text'])
        train_y.append(line['stars'])
    train_y = np.array(train_y).astype(np.int8)

    for line in tqdm(json_reader(test_path)):
        test_corpus.append(line['text'])
        # test_y.append(line['stars'])
    test_y = np.array(test_y).astype(np.int8)

    return train_corpus, train_y, test_corpus, test_y


def _naive_bayes_scikit(train_corpus, train_y, test_corpus, test_y):
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

def _linear_svm_scikit(train_corpus, train_y, test_corpus):
    start = time()
    vectorizer1 = TfidfVectorizer(stop_words='english', analyzer='word')
    X = vectorizer1.fit_transform(train_corpus) 
    vocab = vectorizer1.vocabulary_

    vectorizer2 = TfidfVectorizer(vocabulary=vocab)
    test_X = vectorizer2.fit_transform(test_corpus)
    test_X = (test_X - mean) / std

    svm = LinearSVC(C=1.0, tol=1e-5)
    svm.fit(X, train_y)

    print("time taken to train LinearSVC (SVM) on Scikit : {:.3f} seconds.".format(time()-start))

    preds = svm.predict(test_X)
    return preds
    # mask = preds == test_y
    # return sum(mask) / len(mask)

def _linear_sgd_scikit(train_corpus, train_y, test_corpus, test_y):
    start = time()
    vectorizer1 = TfidfVectorizer(stop_words='english', analyzer='word')
    X = vectorizer1.fit_transform(train_corpus)
    vocab = vectorizer1.vocabulary_

    vectorizer2 = TfidfVectorizer(vocabulary=vocab)
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

    c_values = [1e-5, 1e-3, 1, 5, 10]
    # c_values = [0.1, 0.5, 2.5]
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

    train_path = "col774_yelp_data/train.json"
    test_path = "col774_yelp_data/test.json"
    output_path = "output.txt"

    train_path, test_path, output_path = sys.argv[1:]
    train_corpus, train_y, test_corpus, test_y = _load_data(train_path, test_path)

    # score_naive = _naive_bayes_scikit(train_corpus, train_y, test_corpus, test_y)
    # print("Accuracy for naive bayes is : {:.3f}".format(score_naive * 100))

    predictions = _linear_svm_scikit(train_corpus, train_y, test_corpus, test_y)
    predictions = predictions.astype(np.int8)
    f = open(output_path, "w")
    for i in range(len(predictions)):
        print(predictions[i], file=f)
    f.close()

    # print("Accuracy for liblinear SVM is : {:.3f}".format(score_liblinear * 100))

    # score_clf = _linear_sgd_scikit(train_corpus, train_y, test_corpus, test_y)
    # print("Accuracy for SGD classifier is : {:.3f}".format(score_clf * 100))

    # _tune_hyperparams()

