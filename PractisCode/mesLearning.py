from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

import pandas as pd

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

df = pd.read_csv("dataset/patients.csv")

df["BP"] = df["BP"].fillna(df["BP"].mean())

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

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("X_scaled",X_scaled)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", DecisionTreeClassifier())
])

pipeline.fit(
    X_train,
    y_train
)

prediction = pipeline.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    prediction
)

print("Accuracy:", accuracy * 100,"%")

