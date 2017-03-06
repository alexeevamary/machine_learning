# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 17:18:03 2017

@author: USER
"""

import re
import pandas
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import *
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
path = 'sms/SMSSpamCollection.txt'
messages = pandas.read_csv('sms/SMSSpamCollection.txt', sep='\t',
                           names=["label", "message"])
print(messages.head())
print(messages.groupby('label').describe())
# спама гораздо меньше(747~4825) - выборка не сбалансирована
dummy = ['ham' for i in range(len(messages))]
print(accuracy_score(dummy, messages.label))
print ('{0:.3f}'.format(accuracy_score(dummy, messages.label)))
#вообще мне нравится результат
#но может нужно лучше, или он не годится для определения спам - неспам

messages['label'] = messages['label'].map({'ham': 0, 'spam':1}).astype(int)
index = 0
for i in messages.label:
    index += 1
a = [index for index, item in enumerate(messages.label) if item == 0]
b = [index for index, item in enumerate(messages.label) if item == 1]
all = [i for i in a[:499]] + [k for k in b[:499]]

new_messages = pandas.DataFrame(messages, index = sorted(all)) 
new_messages.head()



#0з всего
bow = CountVectorizer()
bow.fit_transform(new_messages['message'])
# print(bow.vocabulary_)

bow = TfidfVectorizer()
bow.fit_transform(new_messages['message'])
# print(bow.vocabulary_)

bowed_messages = bow.transform(new_messages['message'])
naive_model = MultinomialNB()
naive_model.fit(bowed_messages, new_messages['label'])
cv_results = cross_val_score(naive_model, bowed_messages, new_messages['label'], cv=10, scoring='accuracy')
print(cv_results.mean(), cv_results.std())
#print(np.mean(cross_val_score(naive_model, bowed_messages, new_messages['label'], cv=10, scoring='accuracy')))
# 0,948

#1 без знаков препинания
new_messages['message'] = [re.sub('[!"?/\\().:;,-]', '', msg) for msg in new_messages['message']]

bow = CountVectorizer()
bow.fit_transform(new_messages['message'])
#mean =0,941 std =0,017
bow = TfidfVectorizer()
bow.fit_transform(new_messages['message'])
#mean =0,949 std =0,014
bowed_messages = bow.transform(new_messages['message'])
naive_model = MultinomialNB()
naive_model.fit(bowed_messages, new_messages['label'])
cv_results = cross_val_score(naive_model, bowed_messages, new_messages['label'], cv=10, scoring='accuracy')
print(cv_results.mean(), cv_results.std())
#ненамного лучше оказался CountVectorizer


#2 токенизация
new_messages['message'] = [word_tokenize(msg) for msg in new_messages['message']]
bow = CountVectorizer()
bow.fit_transform(new_messages['message'])
#mean =0,951 std =0,019
bow = TfidfVectorizer()
bow.fit_transform(new_messages['message'])
#mean =0,948 std =0,018
bowed_messages = bow.transform(new_messages['message'])
naive_model = MultinomialNB()
naive_model.fit(bowed_messages, new_messages['label'])
cv_results = cross_val_score(naive_model, bowed_messages, new_messages['label'], cv=10, scoring='accuracy')
print(cv_results.mean(), cv_results.std())
#ненамного лучше оказался CountVectorizer
#понятно, что TfidfVectorizer() оказался хуже, поэтому дальше я его не проверяю даже
#3 лемматизация 
lemmatizer = WordNetLemmatizer()
new_messages['message'] = [word_tokenize(msg) for msg in new_messages['message']]
arr = []
for msg in new_messages['message']:
    msg = [lemmatizer.lemmatize(i.lower()) for i in msg]
    arr.append(' '.join(msg))
new_messages['message'] = arr
new_messages.head()
bow = CountVectorizer()
bow.fit_transform(new_messages['message'])
#mean =0,949 std =0,017
bowed_messages = bow.transform(new_messages['message'])
naive_model = MultinomialNB()
naive_model.fit(bowed_messages, new_messages['label'])
cv_results = cross_val_score(naive_model, bowed_messages, new_messages['label'], cv=10, scoring='accuracy')
print(cv_results.mean(), cv_results.std())


#4 стемматизация
stemmer = SnowballStemmer('english')
arr = []
for msg in new_messages['message']:
    msg = [stemmer.stem(i.lower()) for i in msg]
    arr.append(' '.join(msg))
new_messages['message'] = arr
new_messages.head()
bow = CountVectorizer()
bow.fit_transform(new_messages['message'])
#mean =0,951 std =0,017
bowed_messages = bow.transform(new_messages['message'])
naive_model = MultinomialNB()
naive_model.fit(bowed_messages, new_messages['label'])
cv_results = cross_val_score(naive_model, bowed_messages, new_messages['label'], cv=10, scoring='accuracy')
print(cv_results.mean(), cv_results.std())
#чуть луже lemmatizer, но все равно плохая
#5 стоп-слова
bow = CountVectorizer(stop_words='english')
bow.fit_transform(new_messages['message'])
# mean = 0.935 std = 0.025
bowed_messages = bow.transform(new_messages['message'])
naive_model = MultinomialNB()
naive_model.fit(bowed_messages, new_messages['label'])
cv_results = cross_val_score(naive_model, bowed_messages, new_messages['label'], cv=10, scoring='accuracy')
print(cv_results.mean(), cv_results.std())           

#max_df
bow = CountVectorizer(max_df=.2)
bow.fit_transform(new_messages['message'])           
bowed_messages = bow.transform(new_messages['message'])
naive_model = MultinomialNB()
naive_model.fit(bowed_messages, new_messages['label'])
cv_results = cross_val_score(naive_model, bowed_messages, new_messages['label'], cv=10, scoring='accuracy')
print(cv_results.mean(), cv_results.std())           
# mean=0.950 std=0.019        
            
#лучший получился этот
new_messages['message'] = [re.sub('[!"?/\\().:;,-]', '', msg) for msg in new_messages['message']]
bow = CountVectorizer()
bow.fit_transform(new_messages['message'])
bowed_messages = bow.transform(new_messages['message'])
naive_model = MultinomialNB()
naive_model.fit(bowed_messages, new_messages['label'])
cv_results_naive = cross_val_score(naive_model, bowed_messages, new_messages['label'], cv=10, scoring='accuracy')
print(cv_results_naive.mean(), cv_results_naive.std())
#дерево
tree = DecisionTreeClassifier()
tree.fit(bowed_messages, new_messages['label'])
cv_results_tree = cross_val_score(tree, bowed_messages, new_messages['label'], cv=10, scoring='accuracy')
print(cv_results_tree.mean(), cv_results_tree.std())
#mean =0.899 std=0.031
#чуть хуже байеса
#лес
forest = RandomForestClassifier()
forest.fit(bowed_messages, new_messages['label'])
cv_results_forest = cross_val_score(forest, bowed_messages, new_messages['label'], cv=10, scoring='accuracy')
print(cv_results_forest.mean(), cv_results_forest.std())
#mean =0.917 std =0.021
#хуже байеса и лучше дерева
#

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

X, y = bowed_messages, new_messages['label']


title = "Learning Curve (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = naive_model
plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
#на самом деле я не могу никак прокомментировать график - у меня он не отображается
#к тому же попеременно то работают, то не работают токенизация, лемматизация, стемматизация
# не знаю (я не меняю при этом ничего) почему так происходит.
# на моей выборке получилось в общем, что байес выиграл
# делать с последним пунктом - не знаю

