import nltk
import csv
import string
from symspellpy import SymSpell, Verbosity
import pkg_resources
import pickle

with open('reviews_train.csv') as review_file:
    reader = list(csv.reader(review_file, delimiter=','))
    documents = [[row[0], row[1], row[2]] for row in reader]

result = []
spell = SymSpell(max_dictionary_edit_distance=2,prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

for review in documents:
    for word in review[1].split(' '):
        if(len(word) > 0):
            if review[0].lower() not in word.lower():
                word = word.translate(str.maketrans('', '', string.punctuation))
                try:
                    result.append(str(spell.lookup(word,Verbosity.CLOSEST, max_edit_distance=2)[0]).split(',')[0])
                except IndexError:
                    result.append(word)

all_words = nltk.FreqDist(word.lower() for word in result)
word_features = []
for word in list(all_words)[:200]:
    if all_words[word] > 25:
        word_features.append(word)

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
featuresets = []
for review in documents:
    featuresets.append((document_features(review[1]),review[2]))

classifier = nltk.NaiveBayesClassifier.train(featuresets)
f = open('sentiment_analyzer.pickle', 'wb')
pickle.dump(classifier,f)
f.close()
f = open('word_features.pickle', 'wb')
pickle.dump(word_features,f)
f.close()