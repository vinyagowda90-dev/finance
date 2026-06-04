import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("data/Finance_data.csv")

encoders = {}

categorical = [
    "gender",
    "Objective",
    "Duration",
    "Purpose",
    "Avenue"
]

for col in categorical:

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col])

    encoders[col] = le

X = df.drop("Avenue", axis=1)

y = df["Avenue"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

joblib.dump(rf, "models/random_forest.pkl")
joblib.dump(lr, "models/logistic_regression.pkl")
joblib.dump(dt, "models/decision_tree.pkl")
joblib.dump(encoders, "models/encoders.pkl")

print("Models Saved")
