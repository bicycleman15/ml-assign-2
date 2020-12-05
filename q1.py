import numpy as np
import nltk
from col774_yelp_data.utils import json_reader, getStemmedDocuments
from tqdm import tqdm
from utils import simple_tokenizer, _load_object, calc_accuracy, create_confusion_matrix, plot_confusion_matrix
from utils import take_mean_logits, logits_to_prob_vector, plot_roc_curve, stemmedTokenizer, lemmaTokenizer, lemma_bi_gram_tokenizer
import pickle
from os import path
import os
import sys

class NaiveBayes:
    def __init__(self, name):
        self.dictionary = {}
        self.class_to_word_count = {}
        self.classes = [1, 2, 3, 4, 5]
        self.star_count = np.zeros(len(self.classes))

        self.name = name
        self.load_weights()

    def load_weights(self):
        if not path.exists("weights"):
            print("weights dir does not exist. Creating new one.")
            os.makedirs("weights")

        if path.exists("weights/{}-dict.pickle".format(self.name)):
            self.dictionary = _load_object("weights/{}-dict.pickle".format(self.name))
        
        if path.exists("weights/{}-star-count.pickle".format(self.name)):
            self.star_count = _load_object("weights/{}-star-count.pickle".format(self.name))
        
        if path.exists("weights/{}-model-weight.pickle".format(self.name)):
            self.class_to_word_count = _load_object("weights/{}-model-weight.pickle".format(self.name))

    def create_dict(self, reader, tokenizer):
        
        if len(self.dictionary) > 0:
            print("Loaded cached vocab.")
            return

        print("Creating Vocabulary. Please be patient.")
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
            print("Loaded cached naive bayes model.")
            return
        
        print("Training the model on the formed vocab.")

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
        # gt_labels = []

        print("Running predicition on test data.")

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
            # gt_labels.append(int(line['stars'])-1)
        
        logits = np.array(logits)
        # gt_labels = np.array(gt_labels)

        return logits

if __name__ == '__main__':

    train_path = "col774_yelp_data/train.json"
    test_path = "col774_yelp_data/test.json"
    output_path = "output.txt"

    train_path, test_path, output_path = sys.argv[1:]

    model = NaiveBayes(name="simple-naive-bayes")
    tokenizer = simple_tokenizer

    # Create a vocab
    model.create_dict(json_reader(train_path), tokenizer)

    # Train the model
    model.train(json_reader(train_path), tokenizer)

    # Run the model on test data
    logits = model.predict(json_reader(test_path), tokenizer)
    predictions = np.argmax(logits, axis=1).astype(np.uint8)
    predictions += 1 # add 1 to restore class mappings

    # Dump predictions to output file
    f = open(output_path, "w")
    for i in range(len(predictions)):
        print(predictions[i], file=f)
    f.close()

    # Create confusion matrix
    # conf_matrix = create_confusion_matrix(logits, gt_labels)
    # plot_confusion_matrix(conf_matrix, model.classes, name='conf-matrix-simple-naive')
    
    # Print the accuracy on test data
    # print("Accuracy on test data : {:.3f}.".format(calc_accuracy(logits, gt_labels) * 100))

    # Plot the roc curve
    # probs = logits_to_prob_vector(logits)
    # plot_roc_curve(probs, gt_labels, name='prob-roc-micro')   

    # probs = take_mean_logits(logits)
    # plot_roc_curve(probs, gt_labels, name='logits-roc-micro')   
