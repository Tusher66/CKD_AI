import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


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

# model = XGBClassifier(
#     random_state=42
# )

# model = XGBClassifier(
#     random_state=42
# )

model = GradientBoostingClassifier(
    random_state=42
)

model.fit(X_train,y_train)

pradiction = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    pradiction
)

print("Acuracy",accuracy)