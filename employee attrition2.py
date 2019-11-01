# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 01:20:33 2019

@author: ASUS
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 22:18:22 2019

@author: ASUS
"""
import os

os.chdir('D:')
os.getcwd()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df1 = pd.read_csv('C:/Users/ASUS/Desktop/exist_emp.csv')
df2 = pd.read_csv('C:/Users/ASUS/Desktop/left_emp.csv')

df1['attrition'] = 1
df2['attrition'] = 0

df = pd.concat([df1, df2], axis=0, sort=False)

df.shape

df.describe()
df.columns

#dropping employee IDs as they are insignificant for the model

del df['Emp ID']

dummy1 = pd.get_dummies(df['dept'])
dummy2 = pd.get_dummies(df['salary'])

#one hot encoding dept and salary columns

df = pd.concat([df, dummy1], axis=1)
df = pd.concat([df, dummy2], axis=1)
del df['salary']
del df['dept']

# checking the correlation between the variables (independent and dependent)

plt.figure(figsize=(14,12))
sns.heatmap(df.corr(),linewidths=.1,cmap="YlGnBu", annot=True)
plt.yticks(rotation=0);

#noted from heat map that 'last_evaluation' has maximum correlation with number_project and average_monthly_hours, hence dropping last_evaluation

del df['last_evaluation']

#now average_monthly_hours is also highly correlated with number_projects, hence dropping number_projects

del df['number_project']

#checking outliers

def outliers(y):
#    %matplotlib inline
    plt.boxplot(y)
    plt.show()
    
outliers(df['satisfaction_level'])
outliers(df['average_montly_hours'])

#checking for imbalanced data

sns.set()
ax = sns.countplot(x=df["attrition"], y=None, hue=None, data=df)
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points');

#noted from heatmap that attrition is highly correlated with satisfaction level. Investigating further..
    
#sns.set()
#ax = sns.countplot(x=df["attrition"], y=df['satisfaction_level'].mean(), hue=None, data=df)
#for p in ax.patches:
#    ax.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points');

df_invest = df.groupby(['attrition'])['satisfaction_level'].mean()

#fitting the data into a classification algorithm

X = df.drop('attrition', axis = 1)
Y = df['attrition']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

from sklearn.linear_model import LogisticRegression

cl = LogisticRegression(random_state = 0)
cl.fit(X_train, Y_train)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

clf = clf.fit(X_train,Y_train)

#predicting results

Y_pred = clf.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

cm

# from the confusion matrix it can be inferred that the decission tree classifier is the best model among the three!

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = clf, X = X_train, y = Y_train, cv = 10)
accuracies.mean()
accuracies.std()

# Saving model to disk
pickle.dump(clf, open('employee_attrition2.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('employee_attrition2.pkl','rb'))


