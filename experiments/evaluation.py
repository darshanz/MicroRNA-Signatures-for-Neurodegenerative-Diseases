import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

try:
    from experiments.models import make_classifier, resample_training_fold
except ModuleNotFoundError:
    from models import make_classifier, resample_training_fold


def cross_validated_predictions(
    X,
    y,
    feature_subset,
    classifier_type="RF",
    use_smote=True,
    n_splits=10,
    random_state=42,
):
    X_optimal = X[feature_subset]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    y_true_all = []
    y_pred_all = []
    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_optimal, y), start=1):
        X_train = X_optimal.iloc[train_idx]
        X_test = X_optimal.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        X_train, y_train = resample_training_fold(
            X_train,
            y_train,
            use_smote=use_smote,
            random_state=random_state,
        )

        clf = make_classifier(classifier_type, random_state=random_state)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        fold_scores.append(
            {
                "fold": fold,
                "accuracy": accuracy_score(y_test, y_pred),
                "mcc": matthews_corrcoef(y_test, y_pred),
                "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
            }
        )

    return np.array(y_true_all), np.array(y_pred_all), pd.DataFrame(fold_scores)


def evaluate_feature_subset(
    X,
    y,
    feature_subset,
    label_encoder,
    classifier_type="RF",
    use_smote=True,
    n_splits=10,
    random_state=42,
):
    y_true, y_pred, fold_scores = cross_validated_predictions(
        X,
        y,
        feature_subset,
        classifier_type=classifier_type,
        use_smote=use_smote,
        n_splits=n_splits,
        random_state=random_state,
    )

    final_model = make_classifier(classifier_type, random_state=random_state)
    X_train = X[feature_subset]
    y_train = y
    if use_smote:
        X_train, y_train = resample_training_fold(
            X_train,
            y_train,
            use_smote=True,
            random_state=random_state,
        )
    final_model.fit(X_train, y_train)

    class_report = classification_report(
        y_true,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    importances = getattr(final_model, "feature_importances_", np.zeros(len(feature_subset)))

    return {
        "classifier": final_model,
        "features": feature_subset,
        "fold_scores": fold_scores,
        "cv_accuracy": round(float(fold_scores["accuracy"].mean()), 4),
        "cv_accuracy_std": round(float(fold_scores["accuracy"].std(ddof=0)), 4),
        "cv_mcc": round(float(fold_scores["mcc"].mean()), 4),
        "cv_mcc_std": round(float(fold_scores["mcc"].std(ddof=0)), 4),
        "cv_f1_macro": round(float(fold_scores["f1_macro"].mean()), 4),
        "cv_f1_weighted": round(float(fold_scores["f1_weighted"].mean()), 4),
        "cv_precision_macro": round(
            float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "cv_recall_macro": round(
            float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "feature_importances": {
            feature: round(float(importance), 4)
            for feature, importance in zip(feature_subset, importances)
        },
        "classification_report": class_report,
    }


def incremental_feature_selection(
    X,
    y,
    ranked_features,
    use_smote=True,
    n_splits=10,
    random_state=42,
    desc="Incremental Feature Selection",
):
    results = []

    feature_counts = tqdm(
        range(1, len(ranked_features) + 1),
        desc=desc,
        unit="subset",
    )

    for n_features in feature_counts:
        feature_subset = ranked_features[:n_features]

        row = {"n_features": n_features}
        for classifier_type in ["RF", "DT"]:
            y_true, y_pred, fold_scores = cross_validated_predictions(
                X,
                y,
                feature_subset,
                classifier_type=classifier_type,
                use_smote=use_smote,
                n_splits=n_splits,
                random_state=random_state,
            )

            prefix = "rf" if classifier_type == "RF" else "dt"
            row[f"{prefix}_mcc"] = float(fold_scores["mcc"].mean())
            row[f"{prefix}_acc"] = float(fold_scores["accuracy"].mean())
            row[f"{prefix}_f1_weighted"] = float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            )

        results.append(row)

    return pd.DataFrame(results)


def get_optimal_feature_row(ifs_df, ranked_features, ranking_method, classifier_type):
    prefix = "rf" if classifier_type == "RF" else "dt"
    best_index = ifs_df[f"{prefix}_mcc"].idxmax()
    best_row = ifs_df.loc[best_index]
    optimal_n = int(best_row["n_features"])

    return {
        "ranking_method": ranking_method,
        "classifier": classifier_type,
        "optimal_features": optimal_n,
        "best_mcc": float(best_row[f"{prefix}_mcc"]),
        "best_accuracy": float(best_row[f"{prefix}_acc"]),
        "optimal_feature_names": ranked_features[:optimal_n],
    }


def load_optimal_features(path):
    optimal_df = pd.read_csv(path)
    optimal = {}

    for _, row in optimal_df.iterrows():
        key = (row["ranking_method"], row["classifier"])
        optimal[key] = ast.literal_eval(row["optimal_feature_names"])

    return optimal


def make_summary_table(final_classifiers):
    model_name_labels = {
        "mrmr_rf_smote": "mRMR + RF + SMOTE",
        "mrmr_dt_smote": "mRMR + DT + SMOTE",
        "mcfs_rf_smote": "MCFS + RF + SMOTE",
        "mcfs_dt_smote": "MCFS + DT + SMOTE",
        "mrmr_rf_no_smote": "mRMR + RF + (NO SMOTE)",
        "mrmr_dt_no_smote": "mRMR + DT + (NO SMOTE)",
        "mcfs_rf_no_smote": "MCFS + RF + (NO SMOTE)",
        "mcfs_dt_no_smote": "MCFS + DT + (NO SMOTE)",
    }

    rows = []
    for model_name, results in final_classifiers.items():
        report = results["classification_report"]
        rows.append(
            {
                "Model": model_name_labels[model_name],
                "Features": len(results["features"]),
                "Accuracy": round(float(report["accuracy"]), 4),
                "F1-score(weighted)": round(float(report["weighted avg"]["f1-score"]), 4),
                "MCC": results["cv_mcc"],
                "MCC std": results["cv_mcc_std"],
                "SMOTE": "yes" if "no_smote" not in model_name else "no",
            }
        )

    return pd.DataFrame(rows)


def write_reports(final_classifiers, output_path):
    output_path = Path(output_path)
    report_dict = {}

    for model_name, results in final_classifiers.items():
        report_dict[model_name] = {
            "features": results["features"],
            "cv_accuracy": results["cv_accuracy"],
            "cv_mcc": results["cv_mcc"],
            "classification_report": results["classification_report"],
            "feature_importances": results["feature_importances"],
        }

    output_path.write_text(json.dumps(report_dict, indent=2))
