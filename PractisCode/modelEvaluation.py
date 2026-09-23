import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
from sklearn import tree

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

df = pd.read_csv("dataset/patients.csv")

X = df[
    ["Age",
     "BP",
    "Creatinine"
    ]
]

y = df["CKD"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# model = DecisionTreeClassifier()

model = DecisionTreeClassifier(

    criterion="entropy",

    max_depth=6,

    min_samples_split=5,

    random_state=42

)

model.fit(X_train,y_train)

predictions  = model.predict(X_test)

print("Accuracy")

print(accuracy_score(y_test,predictions))

# print("Confusion Matrix")

# print(confusion_matrix(y_test,predictions))

# print("Classification Report")

# print(classification_report(y_test,predictions))

plt.figure(figsize=(12,8))

tree.plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Healthy","CKD"],
    filled=True
)
