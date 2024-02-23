# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: BHARATHGANESH.S
RegisterNumber:  212222230022
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Read the CSV data
df = pd.read_csv('/content/Book1.csv')

# View the beginning and end of the data
df.head()
df.tail()

# Segregate data into variables
x = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

# Split the data into training and testing sets

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)
# Create a linear regression model
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict values using the model
y_pred = regressor.predict(x_test)

# Display predicted and actual values
print("Predicted values:", y_pred)
print("Actual values:", y_test)

# Visualize the training data
plt.scatter(x_train, y_train, color="black")
plt.plot(x_train, regressor.predict(x_train), color="red")
plt.title("Hours VS scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# Visualize the test data
plt.scatter(x_test, y_test, color="cyan")
plt.plot(x_test, regressor.predict(x_test), color="green")
plt.title("Hours VS scores (Test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
print('MSE = ', mse)

mae = mean_absolute_error(y_test, y_pred)
print('MAE = ', mae)

rmse = np.sqrt(mse)
print('RMSE = ', rmse)
```

## Output:
df.head()


![Screenshot 2024-02-23 093110](https://github.com/bharathganeshsivasankaran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478098/3a496a02-cb98-4631-b9f2-aae2e4979668)


df.tail()

![Screenshot 2024-02-23 093125](https://github.com/bharathganeshsivasankaran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478098/7bbebba8-729f-452e-bf9a-ca1f2b6d4cfa)


Array values of X


![Screenshot 2024-02-23 093142](https://github.com/bharathganeshsivasankaran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478098/d2e21aa7-3bd3-4c77-b404-b65999f98045)


Array values of Y

![Screenshot 2024-02-23 093149](https://github.com/bharathganeshsivasankaran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478098/3e58eea2-eb6d-4f85-8b83-f644d6aeb8f2)

Values of Y Prediction

![Screenshot 2024-02-23 093156](https://github.com/bharathganeshsivasankaran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478098/02580ec5-f3f7-4d73-8f87-614394139b73)


Values of Y test

![Screenshot 2024-02-23 093206](https://github.com/bharathganeshsivasankaran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478098/492f94c9-df9f-4bfc-ba5c-dfe546432897)

Training set graph

![Screenshot 2024-02-23 093332](https://github.com/bharathganeshsivasankaran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478098/fdff8b03-5ceb-493a-a755-86dba0bb578c)


Test set graph

![Screenshot 2024-02-23 093358](https://github.com/bharathganeshsivasankaran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478098/732934ea-4244-4334-aeb7-4a8c61473367)

Values of MSE

![Screenshot 2024-02-23 094057](https://github.com/bharathganeshsivasankaran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478098/cf744e82-f1ce-4963-9834-1d57a6ca2958)


Values of MAE

![Screenshot 2024-02-23 094104](https://github.com/bharathganeshsivasankaran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478098/9fa9ec02-fd8f-45e3-b4c9-396e8eeb2063)

Values of RMSE


![Screenshot 2024-02-23 094116](https://github.com/bharathganeshsivasankaran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478098/08d60c8c-9678-49dd-9ec0-68d9ddf0d4f2)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
