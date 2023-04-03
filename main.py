import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
from nltk.tokenize import word_tokenize, sent_tokenize

from nltk.corpus import stopwords
from string import punctuation # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

from nltk.collocations import *

from nltk.stem.lancaster import LancasterStemmer

from nltk.corpus import wordnet
from nltk.wsd import lesk

text="Anna's passion for aerospace engineering stemmed from her childhood dream of joining the Italian Air Force. This led her to pursue her studies in aerospace engineering and eventually to Airbus, where she currently oversees the landing gear shock absorber for the A321XLR."

sents = sent_tokenize(text)
words = word_tokenize(text);
print(words)
customStopWords = set(stopwords.words('english')+list(punctuation))

filterWords = [word for word in words if word not in customStopWords]
print(filterWords)

bigramMeasures = nltk.collocations.BigramAssocMeasures;
finder = BigramCollocationFinder.from_words(filterWords);
# print(sorted(finder.ngram_fd.items()))

steamWords = [LancasterStemmer().stem(word) for word in filterWords]
print(steamWords)

print(nltk.pos_tag(filterWords))

sense1 = lesk(word_tokenize('Sing in a lower tone,, along with the bass'), 'bass')
print(sense1, sense1.definition())