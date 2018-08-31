import nltk
import random
import string
import pickle
from collections import defaultdict
from nltk.corpus import movie_reviews as mr
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naiave_bayes import MultinomialNB, GaussianNB, BernoulliNB

documents = defaultdict(list)

from nltk.corpus import stopwords
stop = stopwords.words('english')

##one liner of loop below
# documents = [(list(movie_reviews.words(fileid)), catagory)
# 			for catagory in movie_reviews.catagories()
# 			for fileid in movie_reviews.fileids(catagory)]

# documents = []

# for catagory in movie_reviews.catagories():
# 	for fileidin in movie_reviews.fileids(catagory):
# 		document.append(list(movie_reviews.words(fileid)), category)
		

# random.shuffle(documents)

# print(documents[1])

for i in mr.fileids():
    documents[i.split('/')[0]].append(i)

random.shuffle(documents['pos'])
random.shuffle(documents['neg'])      

#print(documents['pos'][:10]) # first ten pos reviews.
#print
#print(documents['neg'][:10]) # first ten neg reviews.


documents = [([w for w in mr.words(i) if w.lower() not in stop and w.lower() not in string.punctuation], i.split('/')[0]) for i in mr.fileids()]

random.shuffle(documents)

allWords = []
for w in mr.words():
	allWords.append(w.lower())

allWords = nltk.FreqDist(allWords)

#print(allWords.most_common(15))
#print(allWords["stupid"])

wordFeatures = list(allWords.keys()) [:3000]

def findFeatures(document):
	words = set(document)
	features = {}
	for w in wordFeatures:
		features[w] = (w in words)

	return features

#print((findFeatures(mr.words('neg/cv000_29416.txt'))))

featureSets = [(findFeatures(rev), category) for (rev, category) in documents]

trainingSet = featureSets[:1900]
testingSet = featureSets[1900:] 

# classifier = nltk.NaiveBayesClassifier.train(trainingSet)

classifier_f = open("naivebayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close


print("Original Naive Bayes Algo accuracy: ", (nltk.classify.accuracy(classifier, testingSet))*100)
classifier.show_most_informative_features(15)

# saveClassifier = open("naivebayes.pickle","wb")
# pickle.dump(classifier, saveClassifier)
# saveClassifier.close()