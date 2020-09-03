import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import nltk
from nltk import NaiveBayesClassifier
from nltk import classify
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
nltk.download('wordnet')

def preprocess(X,y):
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    lemm = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    Xnew = []
    for text in X:
        text = [i.lower() for i in text if i.isalpha() or i == " "]
        newtext = "".join(text)
        newtext = [lemm.lemmatize(i) for i in newtext.split() if i not in stop_words]
        Xnew.append(" ".join(newtext))
    
    return Xnew, y

def bag_of_words(reviews):
    words_dict = dict([r, True] for r in reviews)
    return words_dict

def buildDataset(X,y):
    X, y = preprocess(X,y)

    revs = []
    for i in range(len(X)):
        text = X[i]
        revs.append((bag_of_words(text.split()), labels[y[i]] ))

    train, test = revs[:int(len(revs)*0.8)], revs[int(len(revs)*0.8):]
    
    print('Train Length: {}\nTest Length: {}'.format(len(train), len(test)))
    
    return train, test

data = pd.read_csv('imdb.csv', encoding='latin-1', index_col = 0)
data = data.drop(columns = ['type', 'file'])
data = data[data.label != 'unsup']
data = data.sample(frac=1)

X = data['review'].values
y = data['label'].values

labels = ['neg', 'pos']

train, test = buildDataset(X,y)

clf = NaiveBayesClassifier.train(train)
accuracy = classify.accuracy(clf, test)

print('Accuracy: {}'.format(accuracy*100))

from sklearn.metrics import confusion_matrix as cm
import matplotlib.pyplot as plt

ypred = [clf.classify(t[0]) for t in test]
ytest = y[int(len(y)*0.8):]

matrix = cm(ytest, ypred, labels = ['pos','neg'])

print("Confusion Matrix\n", matrix)
