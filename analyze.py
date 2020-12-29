import requests
from bs4 import BeautifulSoup
from googlesearch import search
import pickle
import time
from symspellpy import SymSpell, Verbosity
import string
import pkg_resources
from nltk.sentiment.vader import SentimentIntensityAnalyzer


found = False
URL = 'https://www.songkick.com/artists'
while not found:
	print('Artist to search for: ')
	artist = str('Songkick' + input())
	for result in search(artist, tld='co.in', num=5, stop=5, pause=1):
		if URL in result:
			found = True
			URL = result
			print('Artist Found!')
			break

page = requests.get(URL)
time.sleep(0.2)
soup = BeautifulSoup(page.content, 'html.parser')
results = soup.find('div',{'class':'artist-reviews'})
reviews = []
index = 0
for review in results.find_all(class_='review-content'):
	temp = review.find_all('p')
	reviews.append('')
	for line in temp[:-2]:
		reviews[index] += line.text + ' '
	index += 1		

f = open('word_features.pickle', 'rb')
word_features = pickle.load(f)
spell = SymSpell(max_dictionary_edit_distance=2,prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
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
f.close()

f = open('sentiment_analyzer.pickle', 'rb')
classifier = pickle.load(f)
sid = SentimentIntensityAnalyzer()
for review in reviews:
	print(classifier.classify(document_features(review)))
	print(sid.polarity_scores(review))

f.close()



