import nltk
import csv
import string
from symspellpy import SymSpell, Verbosity
import pkg_resources
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle

with open('reviews_train.csv') as review_file:
    reader = csv.reader(review_file, delimiter=',')
    documents = [row[0] for row in reader]

result = []
spell = SymSpell(max_dictionary_edit_distance=2,prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

for review in documents:
    for word in review.split(' '):
        if(len(word) > 0):
            word = word.translate(str.maketrans('', '', string.punctuation))
            try:
                result.append(str(spell.lookup(word,Verbosity.CLOSEST, max_edit_distance=2)[0]).split(',')[0])
            except IndexError:
                result.append(word)

all_words = nltk.FreqDist(word.lower() for word in result)
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in document_words:
        word = word.translate(str.maketrans('', '', string.punctuation))
        try:
            word = str(spell.lookup(word,Verbosity.CLOSEST, max_edit_distance=2)[0]).split(',')[0]
        except IndexError:
            word = word
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

#TODO: Add sentiment column for training data
featuresets = [(document_features(d), c) for (d,c) in documents]