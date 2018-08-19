import nltk
from numpy import mean
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC

from nltk.tokenize import word_tokenize
# Medir el tiempo
import timeit


featuresets_f = open("/home/gonzalo/Documentos/Verano Delfin 2018/Algoritmos de prueba/Pruebas 2/featuresets.pickle", "rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()

print(len(featuresets))

training_set = featuresets[:2700] # originalmente esta en 10,000
testing_set = featuresets[2700:3000]

def NaiveBayes():
    return nltk.NaiveBayesClassifier.train(training_set)

print(mean(timeit.repeat(lambda: NaiveBayes(), number=1, repeat = 10)))
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(NaiveBayes(), testing_set)) * 100)

def MultinomialNaiveBayes():
    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set)
    return MNB_classifier

print(mean(timeit.repeat(lambda: MultinomialNaiveBayes(), number=1, repeat = 10)))
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MultinomialNaiveBayes(), testing_set)) * 100)

def BernoulliNaiveBayes():
    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    BernoulliNB_classifier.train(training_set)
    return BernoulliNB_classifier

print(mean(timeit.repeat(lambda: BernoulliNaiveBayes(), number=1, repeat = 10)))
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNaiveBayes(), testing_set)) * 100)

def LogisticRegressionAlgorithm():
    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(training_set)
    return LogisticRegression_classifier

print(mean(timeit.repeat(lambda: LogisticRegressionAlgorithm(), number=1, repeat = 10)))
print("LogisticRegression_classifier accuracy percent:",
      (nltk.classify.accuracy(LogisticRegressionAlgorithm(), testing_set)) * 100)

def StochasticGradientDescent():
    SGDClassifier_classifier = SklearnClassifier(SGDClassifier(max_iter=1000, tol=None))
    SGDClassifier_classifier.train(training_set)
    return SGDClassifier_classifier

print(mean(timeit.repeat(lambda: StochasticGradientDescent(), number=1, repeat = 10)))
print(
"SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(StochasticGradientDescent(), testing_set)) * 100)

def LinearSupportVectorClassification():
    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(training_set)
    return LinearSVC_classifier

print(mean(timeit.repeat(lambda: LinearSupportVectorClassification(), number=1, repeat = 10)))
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSupportVectorClassification(), testing_set)) * 100)

