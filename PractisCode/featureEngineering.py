import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import RFE

df = pd.read_csv("dataset/patients.csv")

# print(df.corr(numeric_only=True))

X = df[
    [
        "Age",
        "BP",
        "Creatinine"
    ]
]

y = df["CKD"]

model = RandomForestClassifier(
    random_state=42
)

model.fit(X,y)

# print(model.feature_importances_)

for feature, importance in zip(
    X.columns,
    model.feature_importances_
):
    print(
        feature,
        "→",
        round(importance, 3)
    )

seceloctor = RFE(
    estimator=model,
    n_features_to_select=3
)

seceloctor.fit(X,y)

print("Selector Column")

print(X.columns[
    seceloctor.support_
])