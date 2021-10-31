
#Spam Analysis 

#Loading the libraries 

import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns 

import warnings
warnings.filterwarnings('ignore')



# Importing the dataset

spambase= pd.read_csv('../spambase.csv')
X = spambase.drop("1",axis=1).values
y = spambase["1"].values


# Splitting the dataset into the Training set and Test set


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=25)

print("===========================================================")
print(X_train)
print("===========================================================")
print(y_train)
print("============================================================")
print(X_test)
print("============================================================")
print(y_test)





# Naive bayes uses distance hence we have to transform X and y 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print("=======================================================")
print(X_train)
print("=======================================================")
print(X_test)





# Previewing the shapes of X-Train and Y-Train 

print("===========================================================")
print(X_train.shape)
print("===========================================================")
print(y_train.shape)
print("===========================================================")
print(X_test.shape)
print("===========================================================")
print(y_test.shape)




# Training the Naive Bayes model on the Training set


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)



# Predicting the Test set results


y_pred = classifier.predict(X_test)
print("===========================================================")
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
print("===========================================================")


# Making the Confusion Matrix


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred)
classification_report =classification_report(y_test, y_pred)
print("===========================================================")
print(cm)
print("===========================================================")
print(classification_report)
print("===========================================================")
accuracy_score(y_test, y_pred)
print(f"Our accuracy score is {accuracy_score(y_test, y_pred)} ")
print("===========================================================")


