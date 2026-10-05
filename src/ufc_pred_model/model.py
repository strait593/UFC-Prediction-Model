"""Shared feature definitions and model construction for bout predictions."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "wins_total_diff",
    "losses_total_diff",
    "age_diff",
    "height_diff",
    "weight_diff",
    "reach_diff",
    "SLpM_total_diff",
    "SApM_total_diff",
    "sig_str_acc_total_diff",
    "td_acc_total_diff",
    "str_def_total_diff",
    "td_def_total_diff",
    "sub_avg_diff",
    "td_avg_diff",
]
CATEGORICAL_FEATURES = ["weight_class", "gender", "r_stance", "b_stance"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "winner"


def build_model() -> Pipeline:
    """Create the preprocessing and logistic-regression prediction pipeline."""
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("classifier", LogisticRegression(max_iter=2_000, class_weight="balanced")),
        ]
    )