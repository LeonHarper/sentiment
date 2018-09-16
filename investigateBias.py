import nltk
import random
import string
import pickle
from collections import defaultdict
from nltk.corpus import movie_reviews as mr
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
	def __init__(self, *classifiers):
		self._classifiers = classifiers

	def classify(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		return mode(votes)

	def confidence(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		choice_votes = votes.count(mode(votes))
		conf = choice_votes / len(votes)
		return conf


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
		

# random.shuffle(documents)f

# print(documents[1])

for i in mr.fileids():
    documents[i.split('/')[0]].append(i)

random.shuffle(documents['pos'])
random.shuffle(documents['neg'])      

#print(documents['pos'][:10]) # first ten pos reviews.
#print
#print(documents['neg'][:10]) # first ten neg reviews.


documents = [([w for w in mr.words(i) if w.lower() not in stop and w.lower() not in string.punctuation], i.split('/')[0]) for i in mr.fileids()]

#random.shuffle(documents)

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

# positive data example
trainingSet = featureSets[:1900]
testingSet = featureSets[1900:] 

# negative data example
trainingSet = featureSets[100:]
testingSet = featureSets[:100] 

# classifier = nltk.NaiveBayesClassifier.train(trainingSet)

classifier_f = open("naivebayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close


# print("Original Naive Bayes Algo accuracy: ", (nltk.classify.accuracy(classifier, testingSet))*100)
# classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(trainingSet)
print("MNB_classsifier Algo accuracy: ", (nltk.classify.accuracy(MNB_classifier, testingSet))*100)

# GaussianNB_classifier = SklearnClassifier(GaussianNB())
# GaussianNB_classifier.train(trainingSet)
# print("GaussianNB_classsifier Algo accuracy: ", (nltk.classify.accuracy(GaussianNB_classifier, testingSet))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(trainingSet)
print("BernoulliNB_classsifier Algo accuracy: ", (nltk.classify.accuracy(BernoulliNB_classifier, testingSet))*100)

# LogisticRegression, SGDClassifier
# SVC, LinearSVC, NuSVC

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(trainingSet)
print("LogisticRegression_classsifier Algo accuracy: ", (nltk.classify.accuracy(LogisticRegression_classifier, testingSet))*100)

# SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
# SGDClassifier_classifier.train(trainingSet)
# print("SGDClassifier_classifier Algo accuracy: ", (nltk.classify.accuracy(SGDClassifier_classifier, testingSet))*100)

# SVC_classifier = SklearnClassifier(SVC())
# SVC_classifier.train(trainingSet)
# print("SVC_classifier Algo accuracy: ", (nltk.classify.accuracy(SVC_classifier, testingSet))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(trainingSet)
print("LinearSVC_classifier Algo accuracy: ", (nltk.classify.accuracy(LinearSVC_classifier, testingSet))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(trainingSet)
print("NuSVC_classifier Algo accuracy: ", (nltk.classify.accuracy(NuSVC_classifier, testingSet))*100)





voted_classifier = VoteClassifier(MNB_classifier,
								  BernoulliNB_classifier,
								  LogisticRegression_classifier,
								  LinearSVC_classifier,
								  NuSVC_classifier )
print("voted_classifier accuracy: ", (nltk.classify.accuracy(voted_classifier, testingSet))*100)

print("Classification: ", voted_classifier.classify(testingSet[1][0]), ". Confidence %: ",voted_classifier.confidence(testingSet[1][0])*100)
print("Classification: ", voted_classifier.classify(testingSet[2][0]), ". Confidence %: ",voted_classifier.confidence(testingSet[2][0])*100)
print("Classification: ", voted_classifier.classify(testingSet[3][0]), ". Confidence %: ",voted_classifier.confidence(testingSet[3][0])*100)
print("Classification: ", voted_classifier.classify(testingSet[4][0]), ". Confidence %: ",voted_classifier.confidence(testingSet[4][0])*100)
print("Classification: ", voted_classifier.classify(testingSet[5][0]), ". Confidence %: ",voted_classifier.confidence(testingSet[5][0])*100)

