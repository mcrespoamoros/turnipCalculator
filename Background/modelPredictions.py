import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_classification
from sklearn.metrics import precision_score, recall_score
from sklearn. metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix

turnip_data = pd.read_csv('../Database/stonksData.csv')

nonLabeled = turnip_data[turnip_data['Pattern'].isnull()]
df = turnip_data[turnip_data['Pattern'].notnull()]

pattern_classes = ['Fluctuating', 'Small Spike', 'Large Spike', 'Decreasing']
pattern_label = 'Pattern'
pattern_feature = list(df.columns[df.columns != 'Pattern'])

# Preparing the data for training and testing
df_X, df_y = df[pattern_feature].values, df[pattern_label].values
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y,
                                                    test_size=0.3,
                                                    random_state=0,
                                                    stratify=df_y)
print('Training Set: %d, Test Set: %d \n' % (X_train.shape[0], X_test.shape[0]))

from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, strategy='mean')

imp.fit(X_train)
X_train = imp.transform(X_train)
X_test = imp.transform(X_test)

""" Nearest neighbors classification"""
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=4)
neigh.fit(X_train, y_train)
neigh_pred = neigh.predict(X_test)
print('Neighbor predicted labels: ', neigh_pred)
print('Neighbor Actual labels:    ', y_test)
print('Neighbor Accuracy: ', accuracy_score(y_test, neigh_pred))
print(classification_report(y_test, neigh_pred))
print("Neighbor Overall Precision:", precision_score(y_test, neigh_pred, average='macro'))
print("Neighbor Overall Recall:", recall_score(y_test, neigh_pred, average='macro'))

# Print the confusion matrix
mcm = confusion_matrix(y_test, neigh_pred)
plt.imshow(mcm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(pattern_classes))
plt.xticks(tick_marks, pattern_classes, rotation=45)
plt.yticks(tick_marks, pattern_classes)
plt.xlabel("Predicted Pattern")
plt.ylabel("Actual Pattern")
plt.show()

""" Random Forest Classifier """
from sklearn.ensemble import RandomForestClassifier

forestClass = RandomForestClassifier(n_estimators=100)
forestClass.fit(X_train, y_train)

forest_pred = forestClass.predict(X_test)
print('Forest Predicted labels: ', forest_pred)
print('Forest Actual labels:    ' ,y_test)

print('Forest Accuracy: ', accuracy_score(y_test, forest_pred))

print(classification_report(y_test, forest_pred))
print("Forest Overall Precision:",precision_score(y_test, forest_pred, average='macro'))
print("Forest Overall Recall:",recall_score(y_test, forest_pred, average='macro'))

# Print the confusion matrix
mcm = confusion_matrix(y_test, forest_pred)
plt.imshow(mcm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(pattern_classes))
plt.xticks(tick_marks, pattern_classes, rotation=45)
plt.yticks(tick_marks, pattern_classes)
plt.xlabel("Predicted Pattern")
plt.ylabel("Actual Pattern")
plt.show()


import joblib

# Save the model as a pickle file
filename = './models/forest_modelJet.pkl'
#joblib.dump(forestClass, filename)

filename = './models/Kneigh_modelJet.pkl'
#joblib.dump(neigh, filename)