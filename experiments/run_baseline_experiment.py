import argparse
import ast
from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    from experiments.data import default_data_dir, load_feature_rankings, load_filtered_dataset, repo_root
    from experiments.evaluation import (
        evaluate_feature_subset,
        get_optimal_feature_row,
        incremental_feature_selection,
        make_summary_table,
        write_reports,
    )
    from experiments.visualization import make_all_plots
except ModuleNotFoundError:
    from data import default_data_dir, load_feature_rankings, load_filtered_dataset, repo_root
    from evaluation import (
        evaluate_feature_subset,
        get_optimal_feature_row,
        incremental_feature_selection,
        make_summary_table,
        write_reports,
    )
    from visualization import make_all_plots


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the baseline miRNA experiment pipeline with leakage-safe SMOTE."
    )
    parser.add_argument("--data-dir", default=str(default_data_dir()))
    parser.add_argument("--output-dir", default=str(repo_root() / "experiments" / "output"))
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--no-smote-in-ifs",
        action="store_true",
        help="Run IFS without SMOTE. Final evaluation still reports both SMOTE and no-SMOTE models.",
    )
    return parser.parse_args()


def run_ifs(X, y, rankings, output_dir, use_smote=True, n_splits=10, random_state=42):
    ifs_tables = {}
    optimal_rows = []

    for ranking_method, features in rankings.items():
        print(f"\nRunning IFS for {ranking_method} ({len(features)} ranked features)")
        ifs_df = incremental_feature_selection(
            X,
            y,
            features,
            use_smote=use_smote,
            n_splits=n_splits,
            random_state=random_state,
            desc=f"IFS {ranking_method}",
        )

        filename = f"{ranking_method.lower()}_ifs_results_fixed_smote.csv"
        ifs_df.to_csv(output_dir / filename, index=False)
        ifs_tables[ranking_method] = ifs_df

        for classifier_type in ["RF", "DT"]:
            row = get_optimal_feature_row(
                ifs_df,
                features,
                ranking_method=ranking_method,
                classifier_type=classifier_type,
            )
            optimal_rows.append(row)
            print(
                f"{ranking_method} + {classifier_type}: "
                f"{row['optimal_features']} features, MCC={row['best_mcc']:.4f}"
            )

    optimal_df = pd.DataFrame(optimal_rows)
    optimal_df.to_csv(output_dir / "optimal_feature_subsets_fixed_smote.csv", index=False)

    return optimal_df, ifs_tables


def run_final_classifiers(X, y, label_encoder, optimal_df, output_dir, n_splits=10, random_state=42):
    final_classifiers = {}
    model_runs = []

    for _, row in optimal_df.iterrows():
        ranking_method = row["ranking_method"]
        classifier_type = row["classifier"]
        features = row["optimal_feature_names"]
        if isinstance(features, str):
            features = ast.literal_eval(features)

        short_rank = ranking_method.lower()
        short_clf = classifier_type.lower()

        for use_smote in [True, False]:
            suffix = "smote" if use_smote else "no_smote"
            model_key = f"{short_rank}_{short_clf}_{suffix}"
            model_runs.append((model_key, classifier_type, use_smote, features))

    for model_key, classifier_type, use_smote, features in tqdm(
        model_runs,
        desc="Final classifiers",
        unit="model",
    ):
        print(f"Evaluating {model_key} with {len(features)} features")
        final_classifiers[model_key] = evaluate_feature_subset(
            X,
            y,
            features,
            label_encoder,
            classifier_type=classifier_type,
            use_smote=use_smote,
            n_splits=n_splits,
            random_state=random_state,
        )

    summary_df = make_summary_table(final_classifiers)
    summary_df.to_csv(output_dir / "classification_summary_fixed_smote.csv", index=False)
    write_reports(final_classifiers, output_dir / "classification_reports_fixed_smote.json")

    return summary_df, final_classifiers


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, label_encoder = load_filtered_dataset(data_dir)
    rankings = load_feature_rankings(data_dir)

    print(f"Loaded expression matrix: {X.shape}")
    print("Label encoding:")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"  {class_name} --- {i}")

    optimal_df, ifs_tables = run_ifs(
        X,
        y,
        rankings,
        output_dir,
        use_smote=not args.no_smote_in_ifs,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )

    summary_df, final_classifiers = run_final_classifiers(
        X,
        y,
        label_encoder,
        optimal_df,
        output_dir,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )

    plot_dir = make_all_plots(ifs_tables, rankings, final_classifiers, output_dir)

    print("\nClassification summary")
    print(summary_df.to_string(index=False))
    print(f"\nSaved outputs to: {output_dir}")
    print(f"Saved plots to: {plot_dir}")


if __name__ == "__main__":
    main()
