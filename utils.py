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
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from functools import lru_cache

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

def plot_confusion_matrix(conf_matrix, labels, name="conf-matrix"):
    ax = plt.subplot()
    sns.heatmap(conf_matrix, annot=True, ax = ax, fmt="d")

    # labels, title and ticks
    ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels);

    plt.savefig("{}.png".format(name))

def logits_to_prob_vector(logits):
    unnorm_probs = np.exp(logits)
    norm_probs = unnorm_probs / np.sum(unnorm_probs)
    return norm_probs

def take_mean_logits(logits):
    for i in range(logits.shape[0]):
        logits[i] -= np.mean(logits[i])
    return logits

def plot_roc_curve(probs, gt_labels, name='roc'):
    labels = label_binarize(gt_labels, classes=[0, 1, 2, 3, 4])

    # class_to_plot = 1
    # fpr, tpr, _ = roc_curve(labels[:][class_to_plot], probs[:][class_to_plot])
    fpr, tpr, _ = roc_curve(labels.ravel(), probs.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.plot(fpr, tpr, label='micro-average ROC curve (area = {:.3f})'.format(roc_auc))
    plt.legend(loc="upper left")

    plt.savefig("{}.jpg".format(name))


en_stop = set(stopwords.words('english'))
p_stemmer = PorterStemmer()
p_stemmer = lru_cache(maxsize=None)(p_stemmer.stem)

lemmatizer = WordNetLemmatizer()
lemmatizer = lru_cache(maxsize=None)(lemmatizer.lemmatize)

def stemmedTokenizer(text : str):
    text = _clean_text(text)
    tokens = nltk.tokenize.word_tokenize(text)
    stopped_tokens = filter(lambda token: token not in en_stop, tokens)
    stemmed_tokens = map(lambda token: p_stemmer(token), stopped_tokens)
    return list(stemmed_tokens)

def lemmaTokenizer(text : str):
    text = _clean_text(text)
    tokens = nltk.tokenize.word_tokenize(text)
    stopped_tokens = filter(lambda token: token not in en_stop, tokens)
    stemmed_tokens = map(lambda token: p_stemmer(token), stopped_tokens)
    lemma_tokens = map(lambda token: lemmatizer(token), stemmed_tokens)
    return list(lemma_tokens)

def lemma_bi_gram_tokenizer(text : str):
    text = _clean_text(text)
    tokens = nltk.tokenize.word_tokenize(text)

    stopped_tokens = filter(lambda token: token not in en_stop, tokens)
    stemmed_tokens = map(lambda token: p_stemmer(token), stopped_tokens)
    lemma_tokens = map(lambda token: lemmatizer(token), stemmed_tokens)

    tokens = list(lemma_tokens)
    tokens = list(nltk.bigrams(tokens))
    
    return tokens