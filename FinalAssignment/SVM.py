"""
Author: Dion Satcher
Date: 3/13/20
Final Assignment
CS445
Student ID: 911832609
"""
import pandas as pd
import sklearn.preprocessing
import sklearn.svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, plot_roc_curve
import matplotlib.pyplot as plt

data = pd.read_csv('spambase.data', header=None, index_col=57)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, data.index.values, stratify=data.index.values,
                                                                            test_size=0.5)


scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), index=y_train)
X_test = pd.DataFrame(scaler.transform(X_test), index=y_test)


clf = sklearn.svm.SVC(kernel='linear').fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('SVM Accuracy: ', accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
plot_roc_curve(clf, X_test, y_test, drop_intermediate=True)
plt.show()