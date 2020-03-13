"""
Author: Dion Satcher
Date: 3/13/20
Final Assignment
CS445
Student ID: 911832609
"""
import pandas as pd
import sklearn
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, precision_score, recall_score, \
    accuracy_score

data = pd.read_csv('spambase.data', header=None, index_col=57)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, data.index.values, stratify=data.index.values,
                                                                            test_size=0.5)
scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), index=y_train)
X_test = pd.DataFrame(scaler.transform(X_test), index=y_test)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

print('Logistic Regression Accuracy: ', accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))


