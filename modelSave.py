import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import StandardScaler

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

model = GradientBoostingClassifier(
    random_state=42
)

model.fit(X_train, y_train)

joblib.dump(
    model,
    "ckd_model.joblib"
)

scaller = StandardScaler()

X_train = scaller.fit_transform(X_train)

X_test = scaller.fit_transform(X_test)

joblib.dump(
    scaller,
    "scaler.joblib"
)