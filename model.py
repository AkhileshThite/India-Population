## Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

"""## Importing the dataset"""

dataset = pd.read_csv('Dataset.csv')
X = dataset.iloc[:, :-1].values  # Independent variable
y = dataset.iloc[:, -1].values  # Dependent variable

y = y.reshape(len(y), 1)

print(X)
print(y)

"""## Splitting the dataset into the Training set and Test set"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)

"""## Training the Simple Linear Regression model on the Training set"""

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

"""## Predicting the Test set results

y_pred = regressor.predict(X_test)

"""
"""## Visualising the Training set results

plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Population vs Year (Training set)')
plt.xlabel('Year')
plt.ylabel('Population (in million)')
plt.show()

"""

"""## Visualising the Test set results

plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Population vs Year (Test set)')
plt.xlabel('Year')
plt.ylabel('Population (in million)')
plt.show()

"""

"""## Enter the year for prediction

n = int(input())
np.set_printoptions(precision=2)
print(regressor.predict([[n]]), "million")

"""

"""## Accuracy

from sklearn.metrics import r2_score
print("Accuracy: {:.2f}%".format(r2_score(y_test, y_pred) * 100))

"""
# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results (read)
model = pickle.load(open('model.pkl','rb'))