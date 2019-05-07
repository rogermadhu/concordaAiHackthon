import os
import re
import string
import collections
import csv
import textblob
import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.classify import MaxentClassifier
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentAnalyzer

def read_test_twitter(path_fileName): #Data/twitter-sentiment-analysis2/test.csv
    ord = []
    with open(path_fileName, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            ord.append(row["SentimentText"])
    return ord

def read_train_twitter(path_fileName): #Data/twitter-sentiment-analysis2/train.csv
    ord = collections.OrderedDict()
    with open(path_fileName, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            ord[row["SentimentText"]] = row["Sentiment"]
    return ord

def readFile(path, filename = ""):
    file_content = open(path+filename,'r', encoding='utf8').read()
    return file_content

def getFilesList(path, ext = 'txt'):
    found = []
    for file in os.listdir(path):
        if(file.endswith('.'+ext)):
            found.append(path+'/'+file)
    return found

def get_dataset(files_list, sentiment = 0, dictionary = None):
    if dictionary is None:
        ord = collections.OrderedDict()

        for file in files_list:
            key = readFile(file)
            value = sentiment
            ord[key] = value

        return ord
    else:
        for file in files_list:
            key = readFile(file)
            value = sentiment
            dictionary[key] = value
        return dictionary

def tokenize_train(dataset):
    ret = []
    for data in dataset:
        temp_tokens = word_tokenize(data[0])
    


def main():

    # # # LOADING TRAIN AND TEST DATA

    # # Train set for Twitter Reading IMDB
    # train_files_list = getFilesList("Data/neg")
    # train_dataset = get_dataset(train_files_list, 0)
    # train_files_list = getFilesList("Data/pos")
    # train_dataset = get_dataset(train_files_list, 1, train_dataset)
    # print("")

    # Train set for CSV Reading TWITTER
    train = read_train_twitter("Data/twitter-sentiment-analysis2/train.csv")
    test = read_test_twitter("Data/twitter-sentiment-analysis2/test.csv")
    print("")
    t = []
    for tr in train:
        t.append([(tr, train[tr])])
    print("")
    
    trainer = NaiveBayesClassifier.train
    sent_analyzer = SentimentAnalyzer()
    training_set = sent_analyzer.apply_features(t)

    test_set = sent_analyzer.apply_features(test)
    print("")
    # # # PREPROCESSING TRAIN AND TEST DATA
        # # Trim data
        #for row in train:


        # # Word Tokenize

        # # Remove Stopwords
        

        # # Remove Special Characters


    # # # FEATURE IDENTIFICATION
        # # extract important keywords
        # # 


    # # # TRAIN CLASSIFIER



    # # # TEST CLASSIFIER



    # # # EVALUATE MODEL

if __name__ == '__main__':
    main()