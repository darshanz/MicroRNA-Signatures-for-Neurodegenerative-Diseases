from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def make_classifier(classifier_type, random_state=42):
    if classifier_type == "RF":
        return RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        )

    if classifier_type == "DT":
        return DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            random_state=random_state,
        )

    raise ValueError("Only 'RF' and 'DT' classifiers are supported.")


def resample_training_fold(X_train, y_train, use_smote, random_state=42):
    if not use_smote:
        return X_train, y_train

    smote = SMOTE(sampling_strategy="auto", random_state=random_state)
    return smote.fit_resample(X_train, y_train)
