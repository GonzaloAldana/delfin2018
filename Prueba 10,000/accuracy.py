import nltk
from numpy import mean
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix  # para la matriz de confusion
from sklearn.model_selection import train_test_split #para partir los sets de datos
from nltk.metrics import ConfusionMatrix

featuresets_f = open("C:/Users/Acer/Desktop/delfin2018/featuresets.pickle",
                     "rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()

train_size= 900 * 10
test_size= 100 * 10

chunk = featuresets[:train_size + test_size]
x = []
y = []
for xs, ys in chunk:
    x.append(xs)
    y.append(ys)
    
train_X = x[:train_size]
val_X = x[train_size:train_size + test_size]
train_y = y[:train_size]
val_y = y[train_size:train_size + test_size]

training_set = featuresets[:train_size]

def NaiveBayes():
    return nltk.NaiveBayesClassifier.train(training_set)

def MultinomialNaiveBayes():
    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set)
    return MNB_classifier

def BernoulliNaiveBayes():
    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    BernoulliNB_classifier.train(training_set)
    return BernoulliNB_classifier
    
def LogisticRegressionAlgorithm():
    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(training_set)
    return LogisticRegression_classifier
    
def StochasticGradientDescent():
    SGDClassifier_classifier = SklearnClassifier(SGDClassifier(max_iter=1000, tol=None))
    SGDClassifier_classifier.train(training_set)
    return SGDClassifier_classifier
    
def LinearSupportVectorClassification():
    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(training_set)
    return LinearSVC_classifier

classifiers = [NaiveBayes(), MultinomialNaiveBayes(), BernoulliNaiveBayes(), LogisticRegressionAlgorithm(), StochasticGradientDescent(), LinearSupportVectorClassification()]

for classifier in classifiers:
    print(classifier)

    resultados = classifier.classify_many(val_X)

    uno, dos = confusion_matrix(val_y, resultados)
    tn, fp = uno
    fn, tp = dos
    print("True Negatives: ", tn)
    print("False Positives: ", fp)
    print("False Negatives: ", fn)
    print("True Positives: ", tp)

    accuracy = (tn+tp)*100/(tp+tn+fp+fn)
    print("Accuracy {:0.2f}%".format(accuracy))

    presicion = tp/(tp+fp)
    print("Presicion {:0.2f}".format(presicion))