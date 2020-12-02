import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn import svm
from sklearn.model_selection import cross_val_score

def _load_all_mnist(path="fashion_mnist", split='val'):
    path = os.path.join(path, split + '.csv')
    data = pd.read_csv(path, header=None)
    data = np.array(data)

    images = data[:, :-1]
    labels = data[:, -1]

    # NOTE : delete this later
    mask = labels <= 3
    images = images[mask]
    labels = labels[mask]
    # -----
    
    images /= 255
    return images, labels

def _train_svm_scikit_k_fold(gamma=0.05, C=1.0):
    """Trains a svm and returns the estimator object"""
    print("Running k-fold cross validation for C={:.3f}.".format(C))
    clf = svm.SVC(C=C, gamma=gamma, decision_function_shape='ovo', verbose=True)
    images, labels = _load_all_mnist(split='train')
    scores = cross_val_score(clf, images, labels, cv=5)
    
    score_k_fold = np.mean(scores)

    clf = svm.SVC(C=C, gamma=gamma, decision_function_shape='ovo', verbose=True)
    clf.fit(images, labels)

    images, labels = _load_all_mnist(split='test')
    preds = clf.predict(images)

    mask = preds == labels
    acc = np.sum(mask) / len(mask)

    return score_k_fold, acc

def _plot_C_curve_k_fold():
    c_values = [1e-5, 1e-3, 1, 5, 10]
    score_pairs = [_train_svm_scikit_k_fold(C=x) for x in c_values]

    c_values = np.array(c_values)
    c_values = np.log10(c_values)

    cross_acc = np.array([x[0] for x in score_pairs])
    test_acc = np.array([x[1] for x in score_pairs])

    # plot graphs here
    print(cross_acc)
    print(test_acc)


_plot_C_curve_k_fold()
