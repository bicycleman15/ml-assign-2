import numpy as np
import nltk
from col774_yelp_data.utils import json_reader, getStemmedDocuments
from tqdm import tqdm
from utils import simple_tokenizer, _load_object, calc_accuracy, create_confusion_matrix, plot_confusion_matrix, logits_to_prob_vector, plot_roc_curve, stemmedTokenizer
import pickle
from os import path

class NaiveBayes:
    def __init__(self):
        self.dictionary = {}
        self.class_to_word_count = {}
        self.classes = [1, 2, 3, 4, 5]
        self.star_count = np.zeros(len(self.classes))

        self.name = "stemmed-naive-bayes"

        self.load_weights()

    def load_weights(self):
        if path.exists("weights/{}-dict.pickle".format(self.name)):
            self.dictionary = _load_object("weights/{}-dict.pickle".format(self.name))
        
        if path.exists("weights/{}-star-count.pickle".format(self.name)):
            self.star_count = _load_object("weights/{}-star-count.pickle".format(self.name))
        
        if path.exists("weights/{}-model-weight.pickle".format(self.name)):
            self.class_to_word_count = _load_object("weights/{}-model-weight.pickle".format(self.name))

    def create_dict(self, reader, tokenizer):
        
        if len(self.dictionary) > 0:
            print("Loaded cached model.")
            return

        idx = 0
        for line in tqdm(reader):
            tokens = tokenizer(line['text'])
            self.star_count[int(line['stars'])-1] += 1
            for x in tokens:
                if x not in self.dictionary:
                    self.dictionary[x] = idx
                    idx += 1
        print("Dictionary created with {} words.".format(len(self.dictionary)))
        
        f = open("weights/{}-dict.pickle".format(self.name), "wb")
        pickle.dump(self.dictionary, f)
        f.close()

        f = open("weights/{}-star-count.pickle".format(self.name), "wb")
        pickle.dump(self.star_count, f)
        f.close()
    
    def train(self, reader, tokenizer):
        assert len(self.dictionary) > 0, "initialize dictionary first"

        if len(self.class_to_word_count) > 0:
            print("Loaded cached model.")
            return

        for i in range(len(self.classes)):
            self.class_to_word_count[i] = np.zeros(len(self.dictionary))

        for line in tqdm(reader):
            tokens = tokenizer(line['text'])
            for x in tokens:
                if x in self.dictionary:
                    self.class_to_word_count[int(line['stars'])-1][self.dictionary[x]] += 1
        print("Naive model trained.")

        f = open("weights/{}-model-weight.pickle".format(self.name), "wb")
        pickle.dump(self.class_to_word_count, f)
        f.close()
    
    def predict(self, reader, tokenizer):
        logits = []
        gt_labels = []

        for line in tqdm(reader):
            tokens = tokenizer(line['text'])
            log_probs = []
            for star in range(len(self.classes)):
                log_sum = 0
                total_words_in_class = np.sum(self.class_to_word_count[star])
                for x in tokens:
                    prob = 0
                    if x in self.dictionary:
                        prob = (self.class_to_word_count[star][self.dictionary[x]] + 1) / (total_words_in_class + len(self.dictionary))
                    else:
                        prob = 1 / (total_words_in_class + len(self.dictionary))
                    log_sum += np.log(prob)
                log_sum += np.log(self.star_count[star] / np.sum(self.star_count))
                log_probs.append(log_sum)
            
            logits.append(log_probs)
            gt_labels.append(int(line['stars'])-1)
        
        logits = np.array(logits)
        gt_labels = np.array(gt_labels)

        return logits, gt_labels

if __name__ == '__main__':
    model = NaiveBayes()
    tokenizer = stemmedTokenizer

    model.create_dict(json_reader("col774_yelp_data/train.json"), tokenizer)
    model.train(json_reader("col774_yelp_data/train.json"), tokenizer)

    # outputs = model.predict(json_reader("col774_yelp_data/test.json"), tokenizer)
    # f = open("outputs_stemmed_test.pickle","wb")
    # pickle.dump(outputs, f)
    # f.close()

    logits, gt_labels = _load_object("outputs_stemmed_test.pickle")
    conf_matrix = create_confusion_matrix(logits, gt_labels)
    
    print(calc_accuracy(logits, gt_labels) * 100)
    print(conf_matrix)

    plot_confusion_matrix(conf_matrix, model.classes)

    probs = logits_to_prob_vector(logits)
    plot_roc_curve(logits, gt_labels)
    



