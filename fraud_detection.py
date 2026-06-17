import pandas as pd
import numpy as np
df = pd.read_csv("creditcard.csv.zip")
print(df.head())
print(df.shape)
df.info()
print("\nDataset shape:")
print(df.shape)

print("\ncolumn names:")
print(df.columns)

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nclass count:")
print(df['Class'].value_counts())

import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='Class', data=df)
plt.title("fraud vs non-fraud transactions")
plt.show()

x=df.drop('Class', axis=1)
y=df['Class']

print("\nFeatures (X) shape:", x.shape)
print("Target (y) shape:", y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("\nTraining set shape:", x_train.shape, y_train.shape)
print("Testing set shape:", x_test.shape, y_test.shape)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20, random_state=42)
model.fit(x_train, y_train)
print("\nModel training completed.")

y_pred = model.predict(x_test)
print("\nPredictions on test set completed:")

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("starting model training..")
model.fit(x_train, y_train)
print("model training completed.")

from sklearn.metrics import classification_report, confusion_matrix

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm=confusion_matrix(y_test, y_pred)
print(cm)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
