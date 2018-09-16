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

#print(allWords.most_common(15))
#print(allWords["stupid"])

wordFeatures = list(all_words.keys()) [:5000]

def findFeatures(document):
	words = word_tokenize(document)
	features = {}
	for w in wordFeatures:
		features[w] = (w in words)

	return features

#print((findFeatures(mr.words('neg/cv000_29416.txt'))))

featureSets = [(findFeatures(rev), category) for (rev, category) in documents]

random.shuffle(featureSets)

# positive data example
trainingSet = featureSets[:10000]
testingSet = featureSets[10000:] 

#code from bias testing set
# negative data exampleb
# trainingSet = featureSets[100:]
# testingSet = featureSets[:100] 

classifier = nltk.NaiveBayesClassifier.train(trainingSet)
classifier.show_most_informative_features(15)


# save_classifier = open("naivebayes_short.pickle","wb")
# pick.dump(classifier, save_classifier)
# save_classifier.close()

# classifier_f = open("naivebayes.pickle","rb")
# classifier = pickle.load(classifier_f)
# classifier_f.close


# print("Original Naive Bayes Algo accuracy: ", (nltk.classify.accuracy(classifier, testingSet))*100)

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

