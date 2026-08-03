import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

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

model = RandomForestClassifier(

    n_estimators=4,

    random_state=42

)

model.fit(

    X_train,

    y_train

)

prediction = model.predict(

    X_test

)

accuracy = accuracy_score(

    y_test,

    prediction

)

print("Accuracy:", accuracy*100,"%")

importent = model.feature_importances_

print("importent",importent)