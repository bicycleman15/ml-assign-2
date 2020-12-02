from col774_yelp_data.utils import json_reader
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from col774_yelp_data.utils import json_reader
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from tqdm import tqdm

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

vectorizer1 = TfidfVectorizer(stop_words='english', analyzer='word')
X = vectorizer1.fit_transform(train_corpus)
# print(vectorizer1.get_feature_names()[101000:101005])
vocab = vectorizer1.vocabulary_

vectorizer2 = TfidfVectorizer(vocabulary=vocab)
test_X = vectorizer2.fit_transform(test_corpus)

clf = MultinomialNB(alpha=1.0)
clf.fit(X, train_y)

preds = clf.predict(test_X)

mask = preds == test_y
print(sum(mask) / len(mask))
