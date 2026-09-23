import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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

scaller = StandardScaler()

X_train = scaller.fit_transform(X_train)

X_test = scaller.fit_transform(X_test)

model = KNeighborsClassifier(
    n_neighbors=5
)

model.fit(
    X_train,
    y_train
)

pradict = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    pradict
)

print("Accuracy:", accuracy * 100)

new_patient = pd.DataFrame({
    "Age":[68],
    "BP":[96],
    "Creatinine":[3.0]
})

new_patient = scaller.transform(new_patient)

pradiction = model.predict(new_patient)

print("new_patient",pradiction)