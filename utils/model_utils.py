import joblib

rf = joblib.load("models/random_forest.pkl")
lr = joblib.load("models/logistic_regression.pkl")
dt = joblib.load("models/decision_tree.pkl")

encoders = joblib.load(
    "models/encoders.pkl"
)

def predict_all(data):

    rf_pred = rf.predict(data)[0]

    lr_pred = lr.predict(data)[0]

    dt_pred = dt.predict(data)[0]

    return {
        "Random Forest": rf_pred,
        "Logistic Regression": lr_pred,
        "Decision Tree": dt_pred
    }
