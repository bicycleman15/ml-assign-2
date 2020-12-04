import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn import svm
from sklearn.model_selection import cross_val_score

from ques2 import _load_all_mnist
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def _train_svm_scikit(split='val'):
    """ Trains svm and returns the predictions as well as gt_labels on `split` dataset.
    """
    # Load the data
    images, labels = _load_all_mnist(split='train')
    print("Now training SVM.")
    import time
    start = time.time()
    clf = svm.SVC(decision_function_shape='ovo', kernel='rbf', C=1.0, gamma=0.05)
    clf.fit(images, labels)
    print("time taken to train : {:.3f} seconds.".format(time.time()-start))

    test_images, test_labels = _load_all_mnist(split=split)
    dec = clf.predict(test_images)

    return dec, test_labels

def _find_accuracy(predictions, gt_labels):
    mask = predictions == gt_labels
    return np.sum(mask) / len(mask)

def _plot_confusion_mnist(predictions, gt_labels, name='scikit'):
    conf_matrix = confusion_matrix(gt_labels, predictions)
    # class_list = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    class_list = list(range(10))

    ax = plt.subplot()
    sns.heatmap(conf_matrix, annot=True, ax = ax, fmt="d")

    # labels, title and ticks
    ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(class_list); ax.yaxis.set_ticklabels(class_list);
    plt.savefig("{}.jpg".format(name))

def _train_svm_scikit_k_fold(gamma=0.05, C=1.0):
    """Trains a svm and returns the estimator object"""
    print("Running k-fold cross validation for C={}. Please be patient.".format(C))
    import time
    start = time.time()
    clf = svm.SVC(C=C, gamma=gamma, decision_function_shape='ovo')
    images, labels = _load_all_mnist(split='train')
    scores = cross_val_score(clf, images, labels, cv=5)
    score_k_fold = np.mean(scores)

    clf = svm.SVC(C=C, gamma=gamma, decision_function_shape='ovo')
    clf.fit(images, labels)
    print("time taken to train : {:.3f} seconds.".format(time.time()-start))
    images, labels = _load_all_mnist(split='test')
    preds = clf.predict(images)

    mask = preds == labels
    acc = np.sum(mask) / len(mask)

    return score_k_fold, acc

def _plot_C_curve_k_fold():
    c_values = [1e-5, 1e-3, 1.0, 5.0, 10.0]
    score_pairs = [_train_svm_scikit_k_fold(C=x) for x in c_values]

    print(score_pairs)

    c_values = np.array(c_values)
    c_values = np.log10(c_values)

    print(c_values)

    cross_acc = np.array([x[0] for x in score_pairs])
    test_acc = np.array([x[1] for x in score_pairs])

    # plot graphs here
    f = open("k-fold.txt","w")
    print(list(c_values), file=f)
    print("cross acc : ", end='')
    print(list(cross_acc), file=f)
    print("test acc : ", end='')
    print(list(test_acc), file=f)
    f.close()

if __name__ == "__main__":

    preds, gts = _train_svm_scikit(split='test')
    test = _find_accuracy(preds, gts)
    print("acc on test scikit: {:.3f}".format(test * 100))

    _plot_confusion_mnist(preds, gts, name='svm-scikit-2-b-ii-conf-matrix')
    # _plot_C_curve_k_fold()