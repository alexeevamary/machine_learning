# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 18:32:46 2017

@author: USER
"""
import pydotplus
import numpy as np
import pandas
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.grid_search import GridSearchCV
import seaborn as sns
from IPython.display import Image


sns.set(style="whitegrid", color_codes=True)
np.random.seed(sum(map(ord, "categorical")))

##нарисовать диаграммы
##1) вероятность выжить для мужчин и женщин
#2) вероятность выжить для пассажиров разных социально-экономических классов (Pclass)
#3) стоимость билета в зависимости от социально-экономического класса.
#Написать, что вам удалось узнать из этих диаграмм
#(например, «для пассажира 1 класса вероятность выжить составила XX% и оказалась выше,
#чем у пассажира 2 класса (YY%)»; чем больше информации вы сможете извлечь из диаграмм,
#тем лучше)
df = pandas.read_csv('C:/Users/USER/Desktop/питон 2016-2017/machine/titanic.csv', index_col='PassengerId')
df.head()
print (sns.barplot(x="Sex", y="Survived", data=df))
# для женщин вероятность выжить оказалсь больше на ~0,55
print (sns.barplot(x="Pclass", y="Survived", data=df))
#для 1ого класса вероятность выжить больше всего
#(на 0,15 больше, чем у 2ого класса, и на 0,375 больше, чем у 3его класса)
print (sns.barplot(x="Pclass", y="Fare", data=df))
#для 1ого класса билеты до 85 доходят,
#для 2ого класса билеты дешевле (20), для 3его ~10

#Нарисовать гистограмму, описывающую среднюю вероятность выжить в зависимости от пола и соц.статуса.
#Снова интерпретировать результаты (Например, какова вероятность выжить женщине из первого класса?).
#Записать любые два утверждения в виде формул (нужно вспомнить, что такое совместная и/или условная вероятность).
print (sns.barplot(x="Sex", y="Survived", hue="Pclass", data=df))
# как и в первой диаграмме - для мужчин вероятность выжить существенно меньше,
# вер-ть для женщин п1ого и 2ого класса составляет ~0,95 и 0,9 соответственно,
# женщины 3его класса выживали с вероятностью 0,5
# для мужчин всех классов вероятность выжить была меньше 0,4
# хотя для мужчин 1ого класса вероятность была самая большая ~0,35
# Утверждение1:P(Survive|Male*1class) > P(Survive|Male*2class) - мужчины первоо класса больше выжили, чем мужчины второго класса
# Утверждение2:P(Survive|Male*1class) < P(Survive|Female*1class) - мужчины первого класса меньше выжили, чем женщины первого класса

    
#Почистите данные так, как считаете нужным (не забывайте про коварную переменную Sex;
#постарайтесь не удалять строки).
#Extra: сможете ли вы использовать не 4 столбца, а больше?
#Например, кажется, что если ребёнок ехал с братом/сестрой, то их не разлучат,
#а посадят вместе в шлюпку, и они выживут...
#df.fillna(value=df, method=None, axis=None, inplace=False, limit=None, downcast=None)   
#pandas.get_dummies(df, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False)
df['Age'].fillna((df['Age'].mean()), inplace=True)

##Разделить данные на обучающую и проверочную выборки (или использовать кросс-валидацию).
#Будем строить дерево решений. Нужно выбрать параметр модели, который, на ваш взгляд,
#может повлиять на результат, и выбрать для него возможные значения.
#Прокомментировать свой выбор. Изменяя в цикле значения параметра,
#посчитать для каждого случая точноть, полноту, F-меру (может быть, другие метрики?).
#Изобразить результаты на диаграмме/-ах. Интерпретировать результаты.
#Нарисовать лучшее дерево.
titanic = pandas.DataFrame(df, columns=['Fare', 'Sex', 'Age', 'Pclass'])
titanic['Sex'] = titanic['Sex'].map({'female': 0, 'male':1}).astype(int)
#f = titanic['Age'].median(skipna=True)

X, y = titanic, df['Survived']
X = pandas.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#X_test.describe()
with open("titanic_tree.dot", 'w') as f:
    
# параметр min_samples_split хоть и с перерывами но при увеличении увеличивает все характеристики
#- и этот параметр больше всего влияет на точность, плоноту и ф-меру
#чуть меньше влияет параметр min_samples_leaf
#пример
    clf = DecisionTreeClassifier(min_samples_split=30)
    gs = GridSearchCV(clf, {'min_samples_leaf': ( .2, .3, .4, .5, .1)})
    
#лучший C = 0.2
    print(gs.best_estimator_)
    y_pred_1 = gs.predict(X_test)
    print(classification_report(y_test, y_pred_1))
    print(np.mean(cross_val_score(gs, X_train, y_train, cv=5)))

    f = tree.export_graphviz(clf, out_file=f, feature_names = X_train.columns)
    f.close()
#    graph = pydotplus.graph_from_dot_data(f) 
#    graph.write_pdf("titanic.pdf")


#Проделать аналогичные операции для модели Random Forest. Сравнить результаты.
with open("titanic_random_tree.dot", 'w') as g:
    model = RandomForestClassifier(criterion='gini', max_depth=None, min_samples_split=30, min_samples_leaf=2, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
#параметр n_estimators хоть и с перерывами (неравномерно), но при увеличении увеличивает полноту, точность и ф-меру
#также как и в предыдущем дереве влияет параметр min_samples_split
# и чуть меньше влияет min_samples_leaf
#например
    gs.fit(np.array(X_train), np.array(y_train))
    gs = GridSearchCV(model, {'n_estimators': list(range(2, 300))})
    print(gs.best_estimator_)
#работает долго очень, но если проверять ручками, то мне понравился результат для 200    
    y_pred_2 = gs.predict(X_test)
    print(classification_report(y_test, y_pred_2))
    g = tree.export_graphviz(gs, out_file=g)

#сравнить модели
# на самом то деле они почти не различаются
scores = {'tree': f1_score(y_test, y_1), 'forest': f1_score(y_test, y_pred_2)}
print (pandas.DataFrame.from_dict(data = scores, orient='index').plot(kind='bar', legend=False))
























