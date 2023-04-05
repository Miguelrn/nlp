import urllib3

from bs4 import BeautifulSoup
# pip install -U scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from collections import defaultdict
from string import punctuation
from heapq import nlargest

from sklearn.neighbors import KNeighborsClassifier


def getAllUrl(url, links):
    http = urllib3.PoolManager()
    response = http.request('GET', url) 
    soup = BeautifulSoup(response.data, 'html.parser')
    for a in soup.find_all('a', attrs={'class','blog-pager-older-link'}):
        try:
            url = a['href']
            title = a['title']
            if(title == 'Older Posts'):
                # print(title, url)
                links.append(url)
                getAllUrl(url, links)
        except:
            title = ''
    return


homePage = 'http://doxydonkey.blogspot.com/'
article = "Quora tests video answers to steal Q&A from YouTube: Newly-minted unicorn Quora has even bigger ambitions than text questions-and-answers. And it’s not going to let video giants or startups disrupt its future. This week Quora began testing video answers, because sometimes it’s a lot easier to show someone how something works, the best way to complete a task, or why one thing is better than another than try to write it out for them. Users in the beta group will be able to record videos on iOS or Android as supplements or complete answers that everyone on Quora can watch. It’s considering allowing video uploads, which might offer more polished content but increase spam concerns. Previously, Quora only let users answer with text, natively hosted photos, links, and embedded videos from platforms like YouTube. Now it’s actively hosting and soliciting video uploads. Quora’s entry into the space could box out younger competitors like Justin Kan’s mobile video Q&A app Whale, and video Ask Me Anything app Yam. These apps are focused entirely on simplifying the process of recording video answers to questions with features like filters to make you look better, and both give creators ways to earn money. But Quora’s 190 million users, $226 million in funding, and 8-year head start give it a big edge. It’s been cautiously curating a network of experts and content, while building a brand name known for quality in contrast to its predecessor Yahoo Answers. Its network effect may be tough to break."
links = ['http://doxydonkey.blogspot.com/search?updated-max=2017-05-23T19:53:00-07:00&max-results=7', 'http://doxydonkey.blogspot.com/search?updated-max=2017-05-14T19:02:00-07:00&max-results=7&start=7&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2017-05-02T19:43:00-07:00&max-results=7&start=14&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2017-04-17T19:26:00-07:00&max-results=7&start=21&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2017-04-10T18:56:00-07:00&max-results=7&start=28&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2017-03-30T19:57:00-07:00&max-results=7&start=35&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2017-03-20T19:47:00-07:00&max-results=7&start=42&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2017-03-02T17:42:00-08:00&max-results=7&start=49&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2017-02-21T19:13:00-08:00&max-results=7&start=56&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2017-02-12T18:34:00-08:00&max-results=7&start=63&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2017-02-01T18:56:00-08:00&max-results=7&start=70&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2017-01-22T18:58:00-08:00&max-results=7&start=77&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2017-01-11T18:09:00-08:00&max-results=7&start=84&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2017-01-02T17:59:00-08:00&max-results=7&start=91&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2016-12-22T18:58:00-08:00&max-results=7&start=98&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2016-12-13T18:57:00-08:00&max-results=7&start=105&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2016-12-04T18:58:00-08:00&max-results=7&start=112&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2016-09-09T07:34:00-07:00&max-results=7&start=119&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2016-08-28T20:08:00-07:00&max-results=7&start=126&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2016-08-17T19:24:00-07:00&max-results=7&start=133&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2016-08-07T20:30:00-07:00&max-results=7&start=140&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2016-07-26T19:55:00-07:00&max-results=7&start=147&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2016-07-17T19:47:00-07:00&max-results=7&start=154&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2016-07-06T19:34:00-07:00&max-results=7&start=161&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2016-06-26T19:36:00-07:00&max-results=7&start=168&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2016-06-15T19:23:00-07:00&max-results=7&start=175&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2016-06-06T18:50:00-07:00&max-results=7&start=182&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2016-05-26T20:08:00-07:00&max-results=7&start=189&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2016-05-17T18:52:00-07:00&max-results=7&start=196&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2016-05-06T19:26:00-07:00&max-results=7&start=203&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2016-04-27T19:03:00-07:00&max-results=7&start=210&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2016-04-19T19:36:00-07:00&max-results=7&start=217&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2016-04-10T19:19:00-07:00&max-results=7&start=224&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2016-03-30T19:12:00-07:00&max-results=7&start=231&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2016-03-20T18:41:00-07:00&max-results=7&start=238&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2016-03-09T18:38:00-08:00&max-results=7&start=245&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2016-02-28T17:47:00-08:00&max-results=7&start=252&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2016-02-17T18:44:00-08:00&max-results=7&start=259&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2016-02-08T18:13:00-08:00&max-results=7&start=266&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2016-01-28T19:45:00-08:00&max-results=7&start=273&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2016-01-18T18:30:00-08:00&max-results=7&start=280&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2016-01-07T19:03:00-08:00&max-results=7&start=287&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-12-28T18:26:00-08:00&max-results=7&start=294&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-12-16T18:24:00-08:00&max-results=7&start=301&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-12-07T18:24:00-08:00&max-results=7&start=308&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-11-26T17:49:00-08:00&max-results=7&start=315&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-11-17T18:18:00-08:00&max-results=7&start=322&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-11-05T20:15:00-08:00&max-results=7&start=329&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-10-27T20:04:00-07:00&max-results=7&start=336&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-10-12T19:45:00-07:00&max-results=7&start=343&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-09-30T19:33:00-07:00&max-results=7&start=350&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-09-20T19:11:00-07:00&max-results=7&start=357&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-09-09T19:32:00-07:00&max-results=7&start=364&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-08-31T19:31:00-07:00&max-results=7&start=371&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-08-20T19:29:00-07:00&max-results=7&start=378&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-08-11T19:32:00-07:00&max-results=7&start=385&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-08-02T19:04:00-07:00&max-results=7&start=392&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-07-22T19:39:00-07:00&max-results=7&start=399&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-07-13T19:38:00-07:00&max-results=7&start=406&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-07-02T21:15:00-07:00&max-results=7&start=413&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-06-23T19:30:00-07:00&max-results=7&start=420&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-06-14T19:36:00-07:00&max-results=7&start=427&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-06-02T19:38:00-07:00&max-results=7&start=434&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-05-24T20:16:00-07:00&max-results=7&start=441&by-date=false', 
'http://doxydonkey.blogspot.com/search?updated-max=2015-05-13T20:18:00-07:00&max-results=7&start=448&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-05-04T20:23:00-07:00&max-results=7&start=455&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-04-23T20:19:00-07:00&max-results=7&start=462&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-04-14T19:40:00-07:00&max-results=7&start=469&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-04-05T20:22:00-07:00&max-results=7&start=476&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-03-24T20:12:00-07:00&max-results=7&start=483&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-03-15T20:41:00-07:00&max-results=7&start=490&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-03-03T19:30:00-08:00&max-results=7&start=497&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-02-22T19:55:00-08:00&max-results=7&start=504&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-02-11T20:02:00-08:00&max-results=7&start=511&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-02-02T19:46:00-08:00&max-results=7&start=518&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-01-22T19:50:00-08:00&max-results=7&start=524&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-01-15T19:17:00-08:00&max-results=7&start=529&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2015-01-06T19:48:00-08:00&max-results=7&start=536&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2014-12-25T21:30:00-08:00&max-results=7&start=543&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2014-12-15T19:24:00-08:00&max-results=7&start=550&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2014-12-05T01:52:00-08:00&max-results=7&start=557&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2014-11-26T01:44:00-08:00&max-results=7&start=564&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2014-11-17T01:41:00-08:00&max-results=7&start=571&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2014-11-06T01:38:00-08:00&max-results=7&start=578&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2014-10-28T01:24:00-07:00&max-results=7&start=585&by-date=false', 'http://doxydonkey.blogspot.com/search?updated-max=2014-10-17T01:20:00-07:00&max-results=7&start=592&by-date=false']

# getAllUrl(homePage, links)  

def getText(url):
    http = urllib3.PoolManager()
    response = http.request('GET', url) 
    soup = BeautifulSoup(response.data, 'html.parser')
    
    posts = ''
    for div in soup.find_all('div', attrs={'class','post-body'}):
        posts += (u"\n".join(t.get_text().strip() for t in div.find_all('li')))

    return posts

post = []     
for link in links:
    post.append(getText(link))
    
vectorizer = TfidfVectorizer(max_df=0.5,min_df=2,stop_words='english')
x = vectorizer.fit_transform(post)

km = KMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=1, verbose=True)
km.fit(x)

np.unique(km.labels_, return_counts=True)

text={}
for i,cluster in enumerate(km.labels_):
    oneDoc = post[i]
    if cluster not in text.keys():
        text[cluster] = oneDoc
    else:
        text[cluster] += oneDoc
        
filter = set(stopwords.words('english')+list(punctuation)+["'s", "billion", "million", "’", "“", "”", "-"])
keywords = {}
counts = {}
for cluster in range(3):
    word_sent = word_tokenize(text[cluster].lower())
    word_sent = [word for word in word_sent if word not in filter]
    freq = FreqDist(word_sent)
    keywords[cluster] = nlargest(100, freq, key=freq.get)
    counts[cluster]=freq
    
unique_keys = {}
for cluster in range(3):
    other_clusters=list(set(range(3))-set([cluster]))
    keys_other_cluster=set(keywords[other_clusters[0]]).union(set(keywords[other_clusters[1]]))
    unique=set(keywords[cluster])-keys_other_cluster
    unique_keys[cluster]=nlargest(10, unique, key=counts[cluster].get)


#k-nearest
classifier = KNeighborsClassifier(n_neighbors=10)
classifier.fit(x, km.labels_)

test = vectorizer.transform([article])


print(unique_keys)
print(classifier.predict(test))
    