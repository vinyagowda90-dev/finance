import pandas as pd

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.model_selection import (
    train_test_split
)


# ==========================
# Load Dataset
# ==========================

def load_dataset(path):

    df = pd.read_csv(path)

    return df


# ==========================
# Missing Values
# ==========================

def handle_missing_values(df):

    df = df.copy()

    for col in df.columns:

        if df[col].dtype == "object":

            df[col].fillna(
                df[col].mode()[0],
                inplace=True
            )

        else:

            df[col].fillna(
                df[col].median(),
                inplace=True
            )

    return df


# ==========================
# Remove Duplicates
# ==========================

def remove_duplicates(df):

    return df.drop_duplicates()


# ==========================
# Label Encoding
# ==========================

def label_encode(df):

    df = df.copy()

    encoders = {}

    object_cols = (
        df.select_dtypes(
            include=["object"]
        ).columns
    )

    for col in object_cols:

        encoder = LabelEncoder()

        df[col] = encoder.fit_transform(
            df[col].astype(str)
        )

        encoders[col] = encoder

    return df, encoders


# ==========================
# One Hot Encoding
# ==========================

def one_hot_encode(df):

    df = pd.get_dummies(
        df,
        drop_first=True
    )

    return df


# ==========================
# Scale Features
# ==========================

def scale_features(X):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


# ==========================
# Prepare Dataset
# ==========================

def prepare_dataset(
    df,
    target_column
):

    df = handle_missing_values(df)

    df = remove_duplicates(df)

    df, encoders = label_encode(df)

    X = df.drop(
        target_column,
        axis=1
    )

    y = df[target_column]

    return X, y, encoders


# ==========================
# Train Test Split
# ==========================

def split_dataset(
    X,
    y,
    test_size=0.2,
    random_state=42
):

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


# ==========================
# Complete Pipeline
# ==========================

def preprocess_pipeline(
    file_path,
    target_column
):

    df = load_dataset(file_path)

    X, y, encoders = prepare_dataset(
        df,
        target_column
    )

    X_scaled, scaler = scale_features(X)

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_dataset(
        X_scaled,
        y
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "encoders": encoders,
        "scaler": scaler
    }
