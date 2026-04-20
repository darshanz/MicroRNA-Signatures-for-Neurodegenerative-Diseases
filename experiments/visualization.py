from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib_venn as venn
import numpy as np
import pandas as pd
import seaborn as sns


MODEL_LABELS = {
    "mrmr_rf_smote": "mRMR + RF + SMOTE",
    "mrmr_dt_smote": "mRMR + DT + SMOTE",
    "mcfs_rf_smote": "MCFS + RF + SMOTE",
    "mcfs_dt_smote": "MCFS + DT + SMOTE",
    "mrmr_rf_no_smote": "mRMR + RF + NO SMOTE",
    "mrmr_dt_no_smote": "mRMR + DT + NO SMOTE",
    "mcfs_rf_no_smote": "MCFS + RF + NO SMOTE",
    "mcfs_dt_no_smote": "MCFS + DT + NO SMOTE",
}


def save_plot(fig, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_ranking_comparison(rankings, output_path):
    mrmr_df = pd.DataFrame(
        {"feature_name": rankings["mRMR"], "mrmr_rank": range(1, len(rankings["mRMR"]) + 1)}
    )
    mcfs_df = pd.DataFrame(
        {"feature_name": rankings["MCFS"], "mcfs_rank": range(1, len(rankings["MCFS"]) + 1)}
    )
    comparison_df = pd.merge(mrmr_df, mcfs_df, on="feature_name", how="outer")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=comparison_df, x="mrmr_rank", y="mcfs_rank", s=70, ax=ax)

    max_rank = max(comparison_df["mrmr_rank"].max(), comparison_df["mcfs_rank"].max())
    ax.plot([0, max_rank], [0, max_rank], color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("mRMR Rank")
    ax.set_ylabel("MCFS Rank")
    ax.set_title("mRMR vs MCFS Feature Ranking")
    save_plot(fig, output_path)

    return comparison_df


def plot_ifs_metric(ifs_tables, metric, classifier_type, output_path):
    prefix = "rf" if classifier_type == "RF" else "dt"
    metric_col = f"{prefix}_{metric}"
    metric_label = "MCC" if metric == "mcc" else "Accuracy"

    rows = []
    for ranking_method, ifs_df in ifs_tables.items():
        for _, row in ifs_df.iterrows():
            rows.append(
                {
                    "Number of Features": row["n_features"],
                    metric_label: row[metric_col],
                    "method": f"{classifier_type} + {ranking_method}",
                }
            )

    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.set_style("whitegrid")
    sns.lineplot(
        data=plot_df,
        x="Number of Features",
        y=metric_label,
        hue="method",
        style="method",
        markers=True,
        linewidth=2,
        dashes=False,
        ax=ax,
    )

    for ranking_method, ifs_df in ifs_tables.items():
        best_index = ifs_df[metric_col].idxmax()
        best_row = ifs_df.loc[best_index]
        ax.axvline(x=best_row["n_features"], linestyle="--", alpha=0.35)
        ax.annotate(
            f"{ranking_method}: {int(best_row['n_features'])}",
            xy=(best_row["n_features"], best_row[metric_col]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
        )

    model_name = "Random Forest" if classifier_type == "RF" else "Decision Tree"
    ax.set_title(f"Incremental Feature Selection ({metric_label}) - {model_name}")
    save_plot(fig, output_path)


def plot_classwise_performance_bar(report_dict, model_name, output_path):
    classes = [cls for cls in report_dict.keys() if cls not in ["accuracy", "macro avg", "weighted avg"]]
    metrics = ["precision", "recall", "f1-score"]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, metric in enumerate(metrics):
        values = [report_dict[cls][metric] for cls in classes]
        bars = ax.bar(x + (width * i), values, width, label=metric, alpha=0.82)
        ax.bar_label(bars, padding=3, fmt="%.2f", fontsize=8)

    ax.set_ylabel("Score")
    ax.set_title(f"Class-wise Performance - {model_name}")
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", ncols=3)
    save_plot(fig, output_path)


def plot_classification_heatmaps(final_classifiers, output_path):
    classes = [
        cls
        for cls in next(iter(final_classifiers.values()))["classification_report"].keys()
        if cls not in ["accuracy", "macro avg", "weighted avg"]
    ]

    rows = []
    for model_key, results in final_classifiers.items():
        report = results["classification_report"]
        for cls in classes:
            rows.append(
                {
                    "Model": MODEL_LABELS.get(model_key, model_key),
                    "Class": cls,
                    "Precision": report[cls]["precision"],
                    "Recall": report[cls]["recall"],
                    "F1-score": report[cls]["f1-score"],
                }
            )

    comparison_df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    for ax, metric in zip(axes, ["Precision", "Recall", "F1-score"]):
        pivot_data = comparison_df.pivot(index="Class", columns="Model", values=metric)
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=0.5,
            vmin=0,
            vmax=1,
            ax=ax,
        )
        ax.set_title(f"{metric} Comparison")
        ax.set_xlabel("")

    fig.suptitle("Multi-model Classification Performance", fontsize=15, fontweight="bold")
    save_plot(fig, output_path)

    return comparison_df


def plot_feature_overlap(mrmr_features, mcfs_features, output_path):
    common_features = set(mrmr_features) & set(mcfs_features)

    fig, ax = plt.subplots(figsize=(8, 5))
    venn_diagram = venn.venn2(
        [set(mrmr_features), set(mcfs_features)],
        set_labels=(f"mRMR (Top {len(mrmr_features)})", f"MCFS (Top {len(mcfs_features)})"),
        ax=ax,
    )

    for patch in venn_diagram.patches:
        if patch:
            patch.set_alpha(0.45)

    if venn_diagram.get_patch_by_id("10"):
        venn_diagram.get_patch_by_id("10").set_color("skyblue")
    if venn_diagram.get_patch_by_id("01"):
        venn_diagram.get_patch_by_id("01").set_color("lightcoral")
    if venn_diagram.get_patch_by_id("11"):
        venn_diagram.get_patch_by_id("11").set_color("plum")

    ax.set_title(f"Feature Overlap Between mRMR and MCFS ({len(common_features)} common)")
    save_plot(fig, output_path)

    return common_features


def plot_feature_importance(final_classifiers, output_path, top_n=6):
    model_combinations = [
        ("mrmr_rf_smote", "mRMR + Random Forest"),
        ("mrmr_dt_smote", "mRMR + Decision Tree"),
        ("mcfs_rf_smote", "MCFS + Random Forest"),
        ("mcfs_dt_smote", "MCFS + Decision Tree"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    global_max = 0
    for model_key, _ in model_combinations:
        if model_key in final_classifiers:
            importances = list(final_classifiers[model_key]["feature_importances"].values())
            global_max = max(global_max, max(importances))

    for ax, (model_key, title) in zip(axes, model_combinations):
        if model_key not in final_classifiers:
            ax.axis("off")
            continue

        results = final_classifiers[model_key]
        importance_items = sorted(
            results["feature_importances"].items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_n]

        features = [item[0] for item in importance_items]
        importances = [item[1] for item in importance_items]
        bars = ax.bar(range(len(features)), importances, color="steelblue", alpha=0.75)
        ax.set_title(title)
        ax.set_ylabel("Importance Score")
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels(features, rotation=45, ha="right")
        ax.set_ylim(0, global_max * 1.15 if global_max else 1)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)

        for bar, importance in zip(bars, importances):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (global_max * 0.015 if global_max else 0.01),
                f"{importance:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    save_plot(fig, output_path)


def make_all_plots(ifs_tables, rankings, final_classifiers, output_dir):
    output_dir = Path(output_dir)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    comparison_df = plot_feature_ranking_comparison(
        rankings,
        plot_dir / "feature_ranking_mrmr_mcfs.png",
    )
    comparison_df.to_csv(output_dir / "feature_ranking_comparison_from_pipeline.csv", index=False)

    plot_ifs_metric(ifs_tables, "mcc", "RF", plot_dir / "ifs_mcc_random_forest.png")
    plot_ifs_metric(ifs_tables, "mcc", "DT", plot_dir / "ifs_mcc_decision_tree.png")
    plot_ifs_metric(ifs_tables, "acc", "RF", plot_dir / "ifs_accuracy_random_forest.png")
    plot_ifs_metric(ifs_tables, "acc", "DT", plot_dir / "ifs_accuracy_decision_tree.png")

    if "mrmr_rf_smote" in final_classifiers:
        plot_classwise_performance_bar(
            final_classifiers["mrmr_rf_smote"]["classification_report"],
            "mRMR + Random Forest + SMOTE",
            plot_dir / "classwise_mrmr_rf_smote.png",
        )
    if "mcfs_rf_smote" in final_classifiers:
        plot_classwise_performance_bar(
            final_classifiers["mcfs_rf_smote"]["classification_report"],
            "MCFS + Random Forest + SMOTE",
            plot_dir / "classwise_mcfs_rf_smote.png",
        )

    plot_classification_heatmaps(final_classifiers, plot_dir / "classification_metric_heatmaps.png")

    mrmr_rf_features = final_classifiers["mrmr_rf_smote"]["features"]
    mcfs_rf_features = final_classifiers["mcfs_rf_smote"]["features"]
    common_features = plot_feature_overlap(
        mrmr_rf_features,
        mcfs_rf_features,
        plot_dir / "feature_overlap_mrmr_mcfs.png",
    )

    pd.DataFrame({"feature_name": sorted(common_features)}).to_csv(
        output_dir / "common_features_mrmr_mcfs.csv",
        index=False,
    )

    plot_feature_importance(final_classifiers, plot_dir / "feature_importance.png")

    return plot_dir
