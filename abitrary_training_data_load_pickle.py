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

from nltk.tokenize import word_tokenize

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

short_pos = open("datasets/positive.txt","r").read()
short_neg = open("datasets/negative.txt","r").read()

documents = []

for r in short_pos.split('\n'):
	documents.append( (r, "pos") )

for r in short_neg.split('\n'):
	documents.append( (r, "neg") )

all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
	all_words.append(w.lower())

for w in short_neg_words:
	all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

wordFeatures = list(all_words.keys()) [:5000]

def findFeatures(document):
	words = word_tokenize(document)
	features = {}
	for w in wordFeatures:
		features[w] = (w in words)

	return features

featureSets = [(findFeatures(rev), category) for (rev, category) in documents]

random.shuffle(featureSets)

# positive data example
trainingSet = featureSets[:10000]
testingSet = featureSets[10000:] 

classifier_f = open("naivebayes_short.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

classifier.show_most_informative_features(15)

classifier_f = open("MNB_classifier_short.pickle", "rb")
MNB_classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f = open("BernoulliNB_classifier_short.pickle", "rb")
BernoulliNB_classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f = open("LogisticRegression_classifier_short.pickle", "rb")
LogisticRegression_classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f = open("LinearSVC_classifier_short.pickle", "rb")
LinearSVC_classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f = open("NuSVC_classifier_short.pickle", "rb")
NuSVC_classifier = pickle.load(classifier_f)
classifier_f.close()

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

