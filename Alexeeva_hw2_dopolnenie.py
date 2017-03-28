# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:39:52 2017

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

#это продолжение второй домашки пункт 3                      
#3 в качестве предикторов конструировать специальные признаки
train, test = train_test_split(messages, test_size=0.2)
#посмотрим длину сообщений
for index, row in train.iterrows():
    train.set_value(index, 'length' , len(row['message']))

#train.head()
train.groupby('label').describe()
#общая длина обычных сообщений больше, чем у спама,
# но в среднем по сообщению длиннее оказывается спам

# вообще все это можно было бы сделать через пандас - но он почему то не работает

#верхний регистр
for index, row in train.iterrows():
    text = row['message']
    n = 0
    for i in text:
        if i.isupper():
            n += 1
        else:
            continue
    train.set_value(index, 'upper' , n)
#train.head()
train.groupby('label').describe()
# как и ожидалось, в среднем в спаме больше верхнего регистра

# количество восклицательных знаков
for index, row in train.iterrows():
    text = row['message']
    n = 0
    for i in text:
        if i == '!':
            n += 1
        else:
            continue
    train.set_value(index, 'punct' , n)
#train.head()
train.groupby('label').describe()
# числа небольшие, т.к. я брала только !, без цифр.
# но все равно есть разницы в спаме и обычных сообщениях
# в спаме в среднем больше !


# модель
X_train = train[['length', 'upper', 'punct']].as_matrix()
y_train = np.array(train['label'])
#т.к. в предыдущих частях получился лучшим классификатором
model = MultinomialNB()
model.fit(X_train, y_train)

for index, row in test.iterrows():
    text = row['message']
    upper_n = 0
    punct_n = 0
    for i in text:
        if i.isupper():
            upper_n += 1
        elif i == '!':
            punct_n += 1
    test.set_value(index, 'length', len(row['message']))
    test.set_value(index, 'upper', upper_n)
    test.set_value(index, 'punct', punct_n)

X_test = test[['length', 'upper', 'punct']].as_matrix()
y_test = np.array(test['label'])
y_pred = model.predict(X_test)
#results = cross_val_score(model, X_test, y_pred, cv=10, scoring='accuracy')
#print(results.mean(), results.std())
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
# вообще показан, неплохой результат, хуже конечно, чем классификаторы из предыдущих пунктов
# но тут еще никак не нормализована несбалансированная выборка, как в первом пункте