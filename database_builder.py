import requests
from bs4 import BeautifulSoup
import csv
import time

URL = 'https://www.songkick.com/leaderboards/popular_artists'
page = requests.get(URL)

soup = BeautifulSoup(page.content, 'html.parser')

results = soup.find('div',{'class':'leaderboard'})

artists = results.find_all(class_='name')

artists = [[artist.find('a').text, artist.find('a')['href']] for artist in artists]


with open('reviews.csv', mode='w', newline='') as review_file:
	review_writer = csv.writer(review_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	for artist in artists[:10]:
		page = requests.get('https://www.songkick.com/' + artist[1])
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
		for review in reviews:
			try:
				if(len(review) > 0):
					review_writer.writerow([artist[0], review])
			except UnicodeEncodeError:
				print('Invalid character detected')
