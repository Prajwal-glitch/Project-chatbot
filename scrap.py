#####	Script for Web Scrapping the WHO frequently asked questions

#####	Essential imports
import requests 
import csv
import re
import pandas as pd
from bs4 import BeautifulSoup



#####	Getting all hyperlinks for FAQs	
link = requests.get("https://www.who.int/emergencies/diseases/novel-coronavirus-2019/question-and-answers-hub")
soup = BeautifulSoup(link.text, 'html.parser')
hlinks = []
print("\nSearching www.WHO.int for FAQs links")
for lns in soup.find_all('a', class_ = 'sf-list-vertical__item'):
	hlinks.append("https://www.who.int"+lns.get('href'))
hlinks.pop(51)


##### 	Getting HTML for hyperlinks
reqs = []
#print(hlinks[51])
print("\n ***** Getting question-answers ***** \n")
for lns in hlinks[:-1]:
	print(lns)
	reqs.append(requests.get(lns))
print("\n====== Done! ======")



##### 	Create a database for question-answers
dataframe = pd.DataFrame()



#####	Preparing the data to be stored in dataframe
for r in reqs:
	soup = BeautifulSoup(r.content, 'html.parser')
	#print(soup.prettify())

	s1 = soup.find_all('span', itemprop ='name')
	questions = []
	for q in s1:
		questions.append(q.getText().strip())


	s2 = soup.find_all('div',itemprop = 'text')
	answers = []
	for a in s2:
		answers.append(a.getText().strip())

	data = {'Questions':questions, 'Answers':answers}
	df = pd.DataFrame(data)
	#print(df)
	dataframe = pd.concat([dataframe,df])
	#dataframe = dataframe.append(df)



#####	Output stored to csv file
dataframe.to_csv('file.csv', index=False)
print("Output saved to file.csv")



