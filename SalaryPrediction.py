import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv("Salary_Data[1].csv")

data

X = data.drop('Salary',axis=1)
y = data['Salary']

from sklearn.preprocessing import LabelEncoder
le_Age = LabelEncoder()
le_Gender = LabelEncoder()
le_Education = LabelEncoder()
le_Job = LabelEncoder()
le_Experience = LabelEncoder()

X['Age_n'] = le_Age.fit_transform(X['Age'])
X['Gender_n'] = le_Gender.fit_transform(X['Gender'])
X['Education_n'] = le_Education.fit_transform(X['Education Level'])
X['Job_n'] = le_Job.fit_transform(X['Job Title'])
X['Experience_n'] = le_Experience.fit_transform(X['Years of Experience'])
X

X = X.drop(['Age'	,'Gender'	,'Education Level'	,'Job Title'	,'Years of Experience'	],axis=1)
X

y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=43)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()

print(y_train.isnull().sum())

X_train_clean = X_train[y_train.notnull()]
y_train_clean = y_train.dropna()

dtc.fit(X_train_clean, y_train_clean)

y_pred = dtc.predict(X_test)
from sklearn.metrics import accuracy_score,f1_score
print("Accuracy Score: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("F1 Score (macro): {:.2f}".format(f1_score(y_test, y_pred, average='macro')))

data.iloc[2]

X.iloc[2]

y.iloc[2]

print(dtc.predict([[12,	0,	0,	131,	10]]))
