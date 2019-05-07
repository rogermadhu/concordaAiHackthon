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
import random

files_pos = os.listdir('Data/pos')
files_pos = [open('Data/pos/'+f, 'r', encoding='utf8').read() for f in files_pos]
files_neg = os.listdir('Data/neg')
files_neg = [open('Data/neg/'+f, 'r', encoding='utf8').read() for f in files_neg]
text_a = 'I hate doing this'
text_b = 'what a beautiful day'
text_c = 'I am enjoying programming'
all_words = []
documents = []
test_sentences = [text_a, text_b, text_c]

from nltk.corpus import stopwords
import re

stop_words = list(set(stopwords.words('english')))

#  j is adject, r is adverb, and v is verb
#allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]

for p in files_pos[:100]:
    
    # create a list of tuples where the first element of each tuple is a review
    # the second element is the label
    documents.append( (p, "pos") )
    
    # remove punctuations
    cleaned = re.sub(r'[^(a-zA-Z)\s]','', p)
    
    # tokenize 
    tokenized = word_tokenize(cleaned)
    
    # remove stopwords 
    stopped = [w for w in tokenized if not w in stop_words]
    
    # parts of speech tagging for each word 
    pos = nltk.pos_tag(stopped)
    
    # make a list of  all adjectives identified by the allowed word types list above
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())
   
for p in files_neg[:100]:
    # create a list of tuples where the first element of each tuple is a review
    # the second element is the label
    documents.append( (p, "neg") )
    
    # remove punctuations
    cleaned = re.sub(r'[^(a-zA-Z)\s]','', p)
    
    # tokenize 
    tokenized = word_tokenize(cleaned)
    
    # remove stopwords 
    stopped = [w for w in tokenized if not w in stop_words]
    
    # parts of speech tagging for each word 
    neg = nltk.pos_tag(stopped)
    
    # make a list of  all adjectives identified by the allowed word types list above
    for w in neg:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())


# creating a frequency distribution of each adjectives.
all_words = nltk.FreqDist(all_words)

# listing the 5000 most frequent words
word_features = list(all_words.keys())[:5000]

# function to create a dictionary of features for each review in the list document.
# The keys are the words in word_features 
# The values of each key are either true or false for wether that feature appears in the review or not

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

# Creating features for each review
featuresets = [
        (find_features(rev), category) for (rev, category) in documents
    ]

# Shuffling the documents 
# random.shuffle(featuresets)

training_set = featuresets[:50]
testing_set = featuresets[50:]
classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)

#classifier.show_most_informative_features(15)