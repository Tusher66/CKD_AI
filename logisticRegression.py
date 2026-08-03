import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("dataset/patients.csv")

X = df[
    [
        "Age",
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

model = LogisticRegression()

model.fit(
    X_train,
    y_train
)

pradiction = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    pradiction
)

print("Accuracy:", accuracy * 100)

new_patient = pd.DataFrame(
    {
        "Age":[68],
        "BP":[95],
        "Creatinine":[1.1]
    }
)

pradiction1 = model.predict(new_patient)

probability = model.predict_proba(new_patient)

print("pradiction1",pradiction1)

print("probability",probability)