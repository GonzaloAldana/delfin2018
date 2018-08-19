import nltk
import pickle
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC

from nltk.tokenize import word_tokenize

featuresets_f = open("/home/gonzalo/Documentos/Verano Delfin 2018/Algoritmos de prueba/Pruebas 2/featuresets.pickle", "rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()

print(len(featuresets))

training_set = featuresets[:900] # originalmente esta en 10,000
testing_set = featuresets[900:1000]


@profile
def inicio():
    print("hola")

@profile
def NaiveBayes(training_set):
    return nltk.NaiveBayesClassifier.train(training_set)

inicio()
NaiveBayes(training_set)

