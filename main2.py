import urllib3

from bs4 import BeautifulSoup
from bs4.element import Comment

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation

from nltk.probability import FreqDist
from heapq import nlargest
from nltk.stem.lancaster import LancasterStemmer
from collections import defaultdict

articleUrl = "https://www.20minutos.es/tecnologia/actualidad/basura-espacial-cuantos-satelites-hay-quien-son-soluciones-5115511/"

http = urllib3.PoolManager()
response = http.request('GET', articleUrl)



def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]', 'section']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.find_all('article', attrs={"class": "article-body"})
    # visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.get_text().strip() for t in texts)

text = text_from_html(response.data)

sentences = sent_tokenize(text)
words = word_tokenize(text.lower())

filter = set(stopwords.words('spanish')+list(punctuation))
filterWords = [LancasterStemmer().stem(word) for word in words if word not in filter]

frecuences = FreqDist(filterWords)
# print(frecuences.most_common())

top = nlargest(10, frecuences, key=frecuences.get)

ranking = defaultdict(int)
# print(top)

for i,sent in enumerate(sentences):
    for w in word_tokenize(sent.lower()):
        if(w in top):
            ranking[i] += frecuences[w]
            
sentIndex = nlargest(4, ranking, key=ranking.get)

print([sentences[index] for index in sorted(sentIndex)])