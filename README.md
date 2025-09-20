# requirements: scikit-learn, xgboost, shap, imbalanced-learn, pandas, numpy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, recall_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import shap
import joblib

# --- 1) Load dataset (example: replace with actual file) ---
# df = pd.read_csv("path/to/dataset.csv")
# Example placeholders:
# X = df.drop(columns=['target'])
# y = df['target']
# For demo, create synthetic:
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=2000, n_features=20, n_informative=8,
                           weights=[0.85,0.15], flip_y=0.02, random_state=42)
X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])

# split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=42)

# --- 2) Preprocessing ---
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
# (If you have categorical features, add them here)
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    # ("cat", cat_transformer, categorical_features)
])

# --- 3) Models and hyperparam grids ---
models = {
    "logreg": (LogisticRegression(max_iter=1000, class_weight="balanced", solver="saga"),
               {"clf__C": np.logspace(-4, 4, 10)}),
    "rf": (RandomForestClassifier(n_jobs=-1, class_weight="balanced", random_state=42),
           {"clf__n_estimators": [100,300], "clf__max_depth":[None,6,10], "clf__min_samples_split":[2,5]}),
    "xgb": (XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_jobs=-1, random_state=42),
            {"clf__n_estimators":[100,300], "clf__max_depth":[3,6], "clf__learning_rate":[0.01,0.1,0.2]})
}

best_models = {}
for name, (estimator, param_grid) in models.items():
    # Use SMOTE inside pipeline if imbalance is significant (optional)
    pipe = ImbPipeline(steps=[
        ("pre", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("clf", estimator)
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(pipe, param_distributions=param_grid,
                                n_iter=10, scoring="roc_auc", n_jobs=-1, cv=cv, random_state=42, verbose=0)
    search.fit(X_train, y_train)
    best_models[name] = search
    print(f"{name} best ROC-AUC (CV): {search.best_score_:.4f}")
    print(f"best params: {search.best_params_}")

# Evaluate on test
for name, search in best_models.items():
    preds_proba = search.predict_proba(X_test)[:,1]
    preds = search.predict(X_test)
    roc = roc_auc_score(y_test, preds_proba)
    prec, recall, _ = precision_recall_curve(y_test, preds_proba)
    pr_auc = auc(recall, prec)
    f1 = f1_score(y_test, preds)
    sens = recall_score(y_test, preds)  # recall
    print(f"\n{name} Test: ROC-AUC={roc:.3f}, PR-AUC={pr_auc:.3f}, F1={f1:.3f}, Sens={sens:.3f}")

# Save the best model (example: best XGB)
best_name = max(best_models.keys(), key=lambda n: best_models[n].best_score_)
joblib.dump(best_models[best_name].best_estimator_, "best_model.pkl")

# SHAP explainability for tree model
best_est = best_models[best_name].best_estimator_.named_steps['clf']
preproc = best_models[best_name].best_estimator_.named_steps['pre']
X_test_pre = preproc.transform(X_test)
explainer = shap.Explainer(best_est)
shap_values = explainer(X_test_pre)
shap.plots.beeswarm(shap_values)
