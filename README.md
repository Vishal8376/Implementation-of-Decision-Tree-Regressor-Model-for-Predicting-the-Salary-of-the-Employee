# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null values using .isnull() function.

3.Import LabelEncoder and encode the dataset.

4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

5.Predict the values of arrays.

6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

7.Predict the values of array.

8.Apply to new unknown values.

## Program:
python
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Vishal S
RegisterNumber:212224040364

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree

data = pd.read_csv("/content/Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data["Position"] = le.fit_transform(data["Position"])

data.head()

x=data[["Position","Level"]]

y=data["Salary"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor,plot_tree

dt=DecisionTreeRegressor()

dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)
from sklearn import metrics

mse = metrics.mean_squared_error(y_test,y_pred)

mse
r2=metrics.r2_score(y_test,y_pred)

r2
dt.predict([[5,6]])

plt.figure(figsize=(20, 8))

plot_tree(dt, feature_names=x.columns, filled=True)

plt.show()
*/


## Output:

![image](https://github.com/user-attachments/assets/40a9db69-15f1-4c1b-a2e5-c305bf0c224a)

![image](https://github.com/user-attachments/assets/526c5276-a0a7-4c4b-bf0a-f9c3550b402c)

![image](https://github.com/user-attachments/assets/1ce439c0-fa04-4d80-a1e2-7d77aee635b9)

![image](https://github.com/user-attachments/assets/e84c319c-88e9-461c-9eac-5478863e75ef)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
