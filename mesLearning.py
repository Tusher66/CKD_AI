from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

import pandas as pd

df = pd.read_csv("dataset/patients.csv")

print(df.head())

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

model = DecisionTreeClassifier()

model.fit(
    X_train,
    y_train
)

predictions = model.predict(X_test)

print("predictions",predictions)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("accuracy",accuracy)

print(
    accuracy * 100
)

new_patient = [
    [
        68,
        96,
        2.9
    ]
]

prediction = model.predict(
    new_patient
)

print("klk",prediction)

# for patient, pred in zip(new_patient, predictions):
#     status = "High CKD Risk" if pred == 1 else "Low CKD Risk"
#     print(f"Patient {patient}: {status}")