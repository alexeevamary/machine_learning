# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 19:34:51 2017

@author: USER
"""

import numpy as np
import pandas
import pylab as pl
import matplotlib.pyplot as plt
from nltk import word_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
# взяла только первый сезон, т.к. мой компьютер барахлит и не справляется, да и проверить данные не так уж и плохо на одном сезоне
# сразу отмечу, в первом сезоне 4 главных героя - Кайл, Картмен, Кенни, Стен,
# если бы я брала другие сезоны, то там появились бы еще герои для анализа типа Баттерса
df = pandas.read_csv("season1.csv")
df.head()
# удалим наны, т.к. их заполнение тут невозможно
df = df.dropna()
# сейчас там реплики всех героев, которые встретились, а нам нужны только 4 главных, создадим датасет из их реплик
heroes = df[(df.Character == 'Kyle')|(df.Character == 'Cartman')|(df.Character == 'Kenny')|(df.Character == 'Stan')]
heroes.groupby('Character').describe()
# датасет несбалансирован - реплик у Кенни существенно меньше, чем у остальных (это и не удивительно)
# возможно это можно будет сгладить используя параметры специальные у классификаторов
X_train, X_test, y_train, y_test = train_test_split(heroes['Line'] , heroes['Character'], test_size=0.2)
# я убирала стоп слова, но оставила обсценные данные (во-первых, у меня итак всего один сезон и мало данных
# во-вторых, там может оказаться что-то полезное)
#посмотрим на классификаторы с матрицами cv и tfidf
cv = CountVectorizer(tokenizer=word_tokenize , stop_words="english")
cv_X_train = cv.fit_transform(X_train)
cv_X_test = cv.transform(X_test)

#Baseline
#ну я взяла самое простое - дамми классификатор 
model1_1 = DummyClassifier()
model1_1.fit(cv_X_train, y_train)
model1_1_y_pred = model1_1.predict(cv_X_test)
np.mean(cross_val_score(model1_1, cv_X_train, y_train, cv=5))
#0.329

#дерево
model1_2 = DecisionTreeClassifier()
#model1_2.fit(cv_X_train, y_train)
#y_pred = model1_2.predict(cv_X_test)
np.mean(cross_val_score(model1_2, cv_X_train, y_train, cv=5))
#0.460 лучше

# лес
model1_3 = RandomForestClassifier()
#model1_3.fit(cv_X_train, y_train)
#y_pred = model1_3.predict(cv_X_test)
np.mean(cross_val_score(model1_3, cv_X_train, y_train, cv=5))
#0.467 еще лучше

# мульти
model1_4 = MultinomialNB()
#model1_4.fit(cv_X_train, y_train)
#y_pred = model1_4.predict(cv_X_test)
np.mean(cross_val_score(model1_4, cv_X_train, y_train, cv=5))
#0.527 еще лучше

#логическая регрессия
model1_5 = LogisticRegression()
#model1_5.fit(cv_X_train, y_train)
#y_pred = model1_5.predict(cv_X_test)
np.mean(cross_val_score(model1_5, cv_X_train, y_train, cv=5))
#0.522 самый хороший


tv = TfidfVectorizer(tokenizer=word_tokenize , stop_words="english")
tv_X_train = tv.fit_transform(X_train)
tv_X_test = tv.transform(X_test)

#
model2_1 = DummyClassifier()
#model2_1.fit(tv_X_train, y_train)
#y_pred = model2_1.predict(tv_X_test)
np.mean(cross_val_score(model2_1, tv_X_train, y_train, cv=5))
#0.311 хуже, чем у cv

#дерево
model2_2 = DecisionTreeClassifier()
#model2_2.fit(tv_X_train, y_train)
#y_pred = model2_2.predict(tv_X_test)
np.mean(cross_val_score(model2_2, tv_X_train, y_train, cv=5))
#0.459 чуть хуже, чем у cv

#лес
model2_3 = RandomForestClassifier()
#model2_3.fit(tv_X_train, y_train)
#y_pred = model2_3.predict(tv_X_test)
np.mean(cross_val_score(model2_3, tv_X_train, y_train, cv=5))
#0.502 хуже чем у cv

#мульти
model2_4 = MultinomialNB()
#model2_4.fit(tv_X_train, y_train)
#y_pred = model2_4.predict(tv_X_test)
np.mean(cross_val_score(model2_4, tv_X_train, y_train, cv=5))
#0.486 совсем плохо

#логическая регрессия
model2_5 = LogisticRegression()
#model2_5.fit(tv_X_train, y_train)
#y_pred = model2_5.predict(tv_X_test)
np.mean(cross_val_score(model2_5, tv_X_train, y_train, cv=5))
#0.521 чуть хуже, чем у cv
#


#лучшим классификатором оказался - LogisticRegression на матрице countvectorizer
# я поигралась с параметрами, чтобы улучшить модель
model1_5 = LogisticRegression()
model1_5.fit(tv_X_train, y_train)
model1_5_y_pred = model1_5.predict(cv_X_test)
np.mean(cross_val_score(model1_5, cv_X_train, y_train, cv=5))
model1_5.coef_
# например, если сгладить выборку с помощью параметра class_weight = 'balanced'
logmodel2 = LogisticRegression(class_weight = 'balanced')
logmodel2.fit(tv_X_train, y_train)
logmodel2_y_pred = logmodel2.predict(cv_X_test)
np.mean(cross_val_score(logmodel2, cv_X_train, y_train, cv=5))
# 0.528
logmodel2.coef_



pl.clf()
plt.figure(figsize=(8,6))


fpr, tpr, thresholds = roc_curve(cv_X_test, model1_5_y_pred)
roc_auc  = auc(fpr, tpr)
pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('logistic regression', roc_auc))

fpr, tpr, thresholds = roc_curve(cv_X_test, model1_1_y_pred)
roc_auc  = auc(fpr, tpr)
pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('dummy classifier', roc_auc))


pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.legend(loc=0, fontsize='small')
pl.show()
