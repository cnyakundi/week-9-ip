#Loading the libraries 

import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns 
import warnings
warnings.filterwarnings('ignore')


#Loading the dataset 

titanic=pd.read_csv('../train (5).csv')

#Printing the columns

print("======================================")

print(titanic.columns)

titanic = titanic.drop(['PassengerId','Name','Ticket','Cabin'], 1)

print("======================================")

titanic.head()

#Dealing with Null Values

def age_approx(cols):
    Age=cols[0]
    Pclass=cols[1]

    if pd.isnull(Age):
        if Pclass==1:
            return 37 

        elif Pclass==2:
            return 29 

        else: 
            return 24 

    else: 
        return Age

titanic['Age'] = titanic[['Age','Pclass']].apply(age_approx,axis=1)
# Training dataset 

# Splitting our dataset

sex = pd.get_dummies(titanic['Sex'],drop_first=True)
embark = pd.get_dummies(titanic['Embarked'],drop_first=True)

titanic.drop(columns=['Sex', 'Embarked'],axis=1, inplace=True)

print(titanic.head())

print("=========================")

train = pd.concat([titanic,sex,embark],axis=1)
print("============================================")
print(train.head())



# Splitting our dataset

X = train.drop("Survived",axis=1)
y = train["Survived"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=25)


# Instanciating a KNeighbors Model 


from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)


# Using our model to make a prediction

y_pred= knn.predict(X_test)


# Evaluating the model

from sklearn.metrics import confusion_matrix, classification_report 
my_confusion_matrix = confusion_matrix(y_test, y_pred)
my_classification_report = classification_report(y_test,y_pred)
print(my_confusion_matrix)
print(my_classification_report)


#Picking optimum K

error_rate = []


for i in range(1,50):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


#Plotting the results 


plt.figure(figsize=(10,6))
plt.plot(range(1,50),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# Retraining the model with best observed K Value


# NOW WITH K=15
knn = KNeighborsClassifier(n_neighbors=15)

knn.fit(X_train,y_train)


pred = knn.predict(X_test)

print('WITH K=15')

print('\n')

print('================================================================')
print(confusion_matrix(y_test,pred))
print('\n')

print("========================")
print("\n")
print(classification_report(y_test,pred))









