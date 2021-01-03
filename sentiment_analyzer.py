import nltk
import csv
import string
from symspellpy import SymSpell, Verbosity
import pkg_resources
import pickle
from nltk.sentiment.vader import SentimentIntensityAnalyzer

with open('reviews_train.csv') as review_file:
    reader = list(csv.reader(review_file, delimiter=','))
    documents = [[row[0], row[1], row[2]] for row in reader]

result = []
spell = SymSpell(max_dictionary_edit_distance=2,prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

negs = ['not', 'no', 'didnt']
for review in documents:
    last_neg = False
    neg_index = 0
    for word in review[1].split(' '):
        if(len(word) > 0):
            if review[0].lower() not in word.lower():
                word = word.translate(str.maketrans('', '', string.punctuation))
                try:
                    word = str(spell.lookup(word,Verbosity.CLOSEST, max_edit_distance=2)[0]).split(',')[0]
                    if word.lower() in negs:
                        last_neg = True
                        if word.lower() == 'not':
                            neg_index = 0
                        elif word.lower() == 'no':
                            neg_index = 1
                        elif word.lower() == 'didnt':
                            neg_index = 2
                        continue
                    if last_neg:
                        if neg_index == 0:
                            word = 'not ' + word
                        if neg_index == 1:
                            word = 'no ' + word
                        if neg_index == 2:
                            word = 'didnt ' + word
                        last_neg = False   
                except IndexError:
                    pass                      
                finally:
                    result.append(word)

all_words = nltk.FreqDist(word.lower() for word in result)
word_features = []
for word in list(all_words):
    if all_words[word] >= 10:
        word_features.append(word)

def document_features(document):
    document_words = document.split(' ')
    feature_words = []
    features = {}
    last_neg = False
    neg_index = 0
    for word in document_words:
        word = word.translate(str.maketrans('', '', string.punctuation))
        try:
            word = str(spell.lookup(word,Verbosity.CLOSEST, max_edit_distance=2)[0]).split(',')[0]
            if word.lower() in negs:
                last_neg = True
                if word.lower() == 'not':
                    neg_index = 0
                elif word.lower() == 'no':
                    neg_index = 1
                elif word.lower() == 'didnt':
                    neg_index = 2
                continue
            if last_neg:
                if neg_index == 0:
                    word = 'not ' + word
                if neg_index == 1:
                    word = 'no ' + word
                if neg_index == 2:
                    word = 'didnt ' + word
                last_neg = False
        except IndexError:
            pass
        finally:
            feature_words.append(word)
    for word in word_features:
        features['contains({})'.format(word)] = (word in feature_words)
    return features

#TODO: Add sentiment column for training data
featuresets = []
for review in documents[:828]:
    featuresets.append((document_features(review[1]),review[2]))

classifier = nltk.NaiveBayesClassifier.train(featuresets[:828])
classifier.show_most_informative_features(5)

def test(classifier):
    sid = SentimentIntensityAnalyzer()
    classify_right = 0 
    sid_right = 0
    for x in range(828,len(documents)):
        result = sid.polarity_scores(documents[x][1])
        if (result['neu'] > 0.5):
            sid_classify = 'neu'
        if(result['pos'] > result['neg']):
            sid_classify = 'pos'
        else:
            sid_classify = 'neg'
        classifier_result = classifier.classify(document_features(documents[x][1]))
        if documents[x][2] == classifier_result:
            classify_right += 1
        elif documents[x][2] == 'neu' or classifier_result == 'neu':
            classify_right += 0.5
        if documents[x][2] == sid_classify:
            sid_right += 1
        elif documents[x][2] == 'neu' or sid_classify == 'neu':
            sid_right += 0.5
    print('Classifier accuray: ' + str(classify_right/(len(documents)-828)) + '\nSID Accuray: ' + str(sid_right/(len(documents)-828)))

test(classifier)    
f = open('sentiment_analyzer.pickle', 'wb')
pickle.dump(classifier,f)
f.close()
f = open('word_features.pickle', 'wb')
pickle.dump(word_features,f)
f.close()