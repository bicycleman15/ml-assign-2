import pickle
import re
import string
import re
import nltk

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, roc_curve, auc
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

regex = re.compile('[%s]' % re.escape(string.punctuation))

def _load_object(path:str):
    """Loads a pickled object util functions"""
    f = open(path, "rb")
    obj = pickle.load(f)
    f.close()
    return obj 

def _clean_text(text : str):
    """Removes puncts and lowers the chars"""
    return regex.sub(' ', text.lower())

def simple_tokenizer(text : str):
    text = _clean_text(text)
    return nltk.tokenize.word_tokenize(text)

def calc_accuracy(logits, gt_labels):
    preds = np.argmax(logits, axis=1)
    mask = (preds == gt_labels)
    return np.sum(mask) / len(mask)

def create_confusion_matrix(logits, gt_labels):
    return confusion_matrix(gt_labels, np.argmax(logits, axis=1))

def plot_confusion_matrix(conf_matrix, labels):
    ax = plt.subplot()
    sns.heatmap(conf_matrix, annot=True, ax = ax, fmt="d")

    # labels, title and ticks
    ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels);

    plt.savefig("conf_matrix.png")

def logits_to_prob_vector(logits):
    unnorm_probs = np.exp(logits)
    return unnorm_probs / np.sum(unnorm_probs)

def plot_roc_curve(logits, gt_labels):
    probs = logits_to_prob_vector(logits)
    labels = label_binarize(gt_labels, classes=[0, 1, 2, 3, 4])

    class_to_plot = 1
    # fpr, tpr, _ = roc_curve(labels[:][class_to_plot], probs[:][class_to_plot])
    fpr, tpr, _ = roc_curve(labels.ravel(), probs.ravel())
    roc_auc = auc(fpr, tpr)

    # import pdb; pdb.set_trace()

    plt.figure()
    plt.plot(fpr, tpr,
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc))

    plt.savefig("roc.jpg")


en_stop = set(stopwords.words('english'))
p_stemmer = PorterStemmer()

def stemmedTokenizer(text : str):
    text = _clean_text(text)
    tokens = nltk.tokenize.word_tokenize(text)
    stopped_tokens = filter(lambda token: token not in en_stop, tokens)
    stemmed_tokens = map(lambda token: p_stemmer.stem(token), stopped_tokens)
    return list(stemmed_tokens)
