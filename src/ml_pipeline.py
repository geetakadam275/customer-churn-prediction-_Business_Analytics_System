
import json
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from config import MODEL_PATH, METRICS_PATH, MODEL_DIR
from src.data_utils import clean_data


def train_models(df):
    # Step 1: Clean the dataset
    df = clean_data(df)

    # Step 2: Prepare the target column
    y = df["Churn"].astype(str).str.strip().str.lower()
    y = y.map({"yes": 1, "no": 0})

    # Remove rows with missing or invalid target values
    valid = y.notna()
    df = df.loc[valid].copy()
    y = y.loc[valid].astype(int)

    # Step 3: Separate features and target
    drop_columns = [
        col for col in ["Churn", "customerID"]
        if col in df.columns
    ]

    X = df.drop(columns=drop_columns)

    # Step 4: Identify categorical and numerical columns
    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_features = [
        col for col in X.columns
        if col not in categorical_features
    ]

    # Step 5: Create numerical preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Step 6: Create categorical preprocessing pipeline
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Step 7: Combine preprocessing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Step 8: Split data into training and testing sets
    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Step 9: Define machine learning models
    estimators = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            random_state=42,
            class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            random_state=42,
            class_weight="balanced"
        )
    }

    # Step 10: Train and evaluate each model
    results = {}
    fitted = {}

    for name, estimator in estimators.items():

        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])

        pipe.fit(Xtr, ytr)

        pred = pipe.predict(Xte)

        results[name] = {
            "accuracy": round(
                float(accuracy_score(yte, pred)), 4
            ),
            "precision": round(
                float(precision_score(
                    yte, pred, zero_division=0
                )), 4
            ),
            "recall": round(
                float(recall_score(
                    yte, pred, zero_division=0
                )), 4
            ),
            "f1_score": round(
                float(f1_score(
                    yte, pred, zero_division=0
                )), 4
            ),
            "confusion_matrix": confusion_matrix(
                yte, pred, labels=[0, 1]
            ).tolist()
        }

        fitted[name] = pipe

    # Step 11: Select the model with the highest F1 score
    best = max(
        results,
        key=lambda name: results[name]["f1_score"]
    )

    # Step 12: Save the best model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": fitted[best],
            "features": X.columns.tolist(),
            "name": best
        },
        MODEL_PATH
    )

    # Step 13: Save evaluation metrics
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model": best,
                "models": results
            },
            f,
            indent=2
        )

    # Step 14: Return the results
    return results, best, fitted[best], X.columns.tolist()
