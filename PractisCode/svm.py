import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV


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

# Feature Scaling (SVM-এর জন্য খুব গুরুত্বপূর্ণ)

scaller = StandardScaler()

X_train = scaller.fit_transform(X_train)
X_test = scaller.transform(X_test)

base_model = SVC(
    kernel="sigmoid",
    random_state=42
)

model = CalibratedClassifierCV(
    base_model,
    ensemble=False
)

# Train
model.fit(X_train, y_train)

pradiction = model.predict(X_test)

accuracy = accuracy_score(y_test,pradiction)

print("Accuracy:", accuracy * 100)

# Probability
print("Probability",model.predict_proba(X_test))