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


documents_f = open("pickles/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()

wordFeatures_f = open("pickles/wordFeatures.pickle", "rb")
wordFeatures = pickle.load(wordFeatures_f)
wordFeatures_f.close()

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

open_file = open("pickles/naivebayes_short.pickle", "rb")
naivebayes_short = pickle.load(open_file)
open_file.close()

open_file = open("pickles/MNB_classifier_short.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickles/BernoulliNB_classifier_short.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickles/LogisticRegression_classifier_short.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickles/LinearSVC_classifier_short.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickles/NuSVC_classifier_short.pickle", "rb")
NuSVC_classifier = pickle.load(open_file)
open_file.close()

voted_classifier = VoteClassifier(MNB_classifier,
								  BernoulliNB_classifier,
								  LogisticRegression_classifier,
								  LinearSVC_classifier,
								  NuSVC_classifier )


def sentiment(text):
	feats = findFeatures(text)
	return voted_classifier.classify(feats),voted_classifier.confidence(feats)
