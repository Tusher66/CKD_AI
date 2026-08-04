import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV

df = pd.read_csv("dataset/patients.csv")

X = df[
    [
        "Age",
        "BP",
        "Creatinine"
    ]
]

y = df["CKD"]

params = {
    "max_depth":[2,3,4,5],
    "min_samples_split":[2,5,10]
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    params,
    cv=5
)

grid.fit(X,y)

print("Best Parameter")
print(grid.best_params_)
print("bast accricy")
print(grid.best_score_)