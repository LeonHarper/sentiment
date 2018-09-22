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

all_words = []
documents = []

allowed_word_types = ["J"]

for p in short_pos.split('\n'):
	documents.append( (p, "pos") )
	words = word_tokenize(p)
	pos = nltk.pos_tag(words)
	for w in pos:
		if w[1][0] in allowed_word_types:
			all_words.append(w[0].lower())

for p in short_neg.split('\n'):
	documents.append( (p, "neg") )
	words = word_tokenize(p)
	pos = nltk.pos_tag(words)
	for w in pos:
		if w[1][0] in allowed_word_types:
			all_words.append(w[0].lower())

save_documents = open("pickles/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()

all_words = nltk.FreqDist(all_words)

wordFeatures = list(all_words.keys()) [:5000]

save_word_features = open("pickles/wordFeatures.pickle","wb")
pickle.dump(wordFeatures, save_word_features)
save_word_features.close()

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

classifier = nltk.NaiveBayesClassifier.train(trainingSet)
classifier.show_most_informative_features(15)


save_classifier = open("pickles/naivebayes_short.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(trainingSet)
print("MNB_classsifier Algo accuracy: ", (nltk.classify.accuracy(MNB_classifier, testingSet))*100)

save_classifier = open("pickles/MNB_classifier_short.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(trainingSet)
print("BernoulliNB_classsifier Algo accuracy: ", (nltk.classify.accuracy(BernoulliNB_classifier, testingSet))*100)

save_classifier = open("pickles/BernoulliNB_classifier_short.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(trainingSet)
print("LogisticRegression_classsifier Algo accuracy: ", (nltk.classify.accuracy(LogisticRegression_classifier, testingSet))*100)

save_classifier = open("pickles/LogisticRegression_classifier_short.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(trainingSet)
print("LinearSVC_classifier Algo accuracy: ", (nltk.classify.accuracy(LinearSVC_classifier, testingSet))*100)

save_classifier = open("pickles/LinearSVC_classifier_short.pickle","wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(trainingSet)
print("NuSVC_classifier Algo accuracy: ", (nltk.classify.accuracy(NuSVC_classifier, testingSet))*100)

save_classifier = open("pickles/NuSVC_classifier_short.pickle","wb")
pickle.dump(NuSVC_classifier, save_classifier)
save_classifier.close()

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

