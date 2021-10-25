# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data collection and processing:
sonar_data = pd.read_csv('/content/sonar.all-data (1).csv', header=None) # header=None
sonar_data.head()

# Number of rows and columns
sonar_data.shape

# Mean, STD, data - describes statistical measures of the data. 
sonar_data.describe()

# Values of column '61' - allows us to see how many were mines, how many were rocks.
sonar_data[60].value_counts()

sonar_data.groupby(60).mean()

# Separating data and labels 
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

# Training and test data:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=1)

print(X.shape, X_train.shape, X_test.shape)

# Training our data (model training):
# Logistic regression model. 
model = LogisticRegression()
model.fit(X_train, Y_train) # train on this data

# Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
training_data_accuracy

# Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
test_data_accuracy

# Designing a Predictive System:
input_data = (0.0032,0.0232,0.0012,0.0001,0.0075,0.0032,0.0232,0.0012,0.0001,0.0075,0.0032,0.0232,0.0012,0.0001,0.0075,0.0032,0.0232,0.0012,0.0001,0.0075,0.0032,0.0232,0.0012,0.0001,0.0075,0.0032,0.0232,0.0012,0.0001,0.0075,0.0032,0.0232,0.0012,0.0001,0.0075,0.0032,0.0232,0.0012,0.0001,0.0075,0.0032,0.0232,0.0012,0.0001,0.0075,0.0032,0.0232,0.0012,0.0001,0.0075,0.0032,0.0232,0.0012,0.0001,0.0075,0.0032,0.0232,0.0012,0.0001,0.0075) # Include the sonar data. 

# Change input_data to a NumPy array
input_data_numpy = np.asarray(input_data)

# Re-shape the array
input_data_reshaped = input_data_numpy.reshape(1, -1)

# Returns the prediction based on the data you give it.
prediction = model.predict(input_data_reshaped)
if (prediction[0]=='R'):
  print('The object is a rock.')
else:
  print('The object is a mine.')