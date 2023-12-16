import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


path = 'dataset/kidney_disease.csv'
df = pd.read_csv(path)
print(df.head())
print(df.shape)
df.describe()
print(df['classification'].value_counts())
columns_to_drop = ['classification', 'rbc', 'sod','pot','pcc','ba','cad','dm','htn','id']
X= df.drop(columns=columns_to_drop)
Y = df['classification']
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)

input_data = (40,66,1.02,2.0,4.0,1,120,56,4.3,11.2,32,6500,4.3,1,0,1)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = classifier.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person do not have a kidney disease')
else:
  print('The person have a kidney disease')


import pickle
filename = 't_model.sav'
pickle.dump(classifier, open(filename, 'wb'))
# loading the saved model
loaded_model = pickle.load(open('t_model.sav', 'rb'))
input_data = (40,66,1.02,2.0,4.0,1,120,56,4.3,11.2,32,6500,4.3,1,0,1)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = classifier.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person do not have a kidney disease')
else:
  print('The person have a kidney disease')