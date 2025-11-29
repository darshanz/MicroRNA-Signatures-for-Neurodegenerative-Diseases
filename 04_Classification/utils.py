from pathlib import Path
import matplotlib_venn as venn
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import ast
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, matthews_corrcoef, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
from imblearn.over_sampling import SMOTE

class OptimalFeatures:
    def __init__(self):
        self.data_dir = f"{Path.cwd().parent.parent}/data"
        self.optimal_ftrs_df = pd.read_csv(f'{self.data_dir}/optimal_feature_subsets_ifs.csv')

    def get_optimal_counts(self, method_name, model_type):
        opt_ftrs = self.optimal_ftrs_df[self.optimal_ftrs_df['ranking_method'] == method_name]
        return opt_ftrs[opt_ftrs['classifier'] == model_type]['optimal_features'].values[0]
    
    def get_df(self):
        return self.optimal_ftrs_df
    
    def get_ftr_names(self, method_name, model_type):
        opt_ftrs = self.optimal_ftrs_df[self.optimal_ftrs_df['ranking_method'] == method_name]
        return list(opt_ftrs[opt_ftrs['classifier'] == model_type]['optimal_feature_names'].apply(ast.literal_eval).values)[0]
    




def build_classifier(X, y, feature_subset, label_encoder, classifier_type='RF', use_smote=True):
    X_optimal = X[feature_subset]

    if use_smote:
        smote = SMOTE(sampling_strategy='auto', random_state=42)   
        X_resampled, y_resampled = smote.fit_resample(X_optimal, y)
    else:
        X_resampled, y_resampled = X_optimal, y

    if classifier_type == 'RF':
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        classifier = DecisionTreeClassifier(
            max_depth=5, 
            min_samples_split=10, 
            random_state=42
        )

    scoring = {
        'accuracy': 'accuracy',
        'mcc': make_scorer(matthews_corrcoef),
        'f1_macro': 'f1_macro',
        'f1_weighted': 'f1_weighted',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro'
    }


    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_results = cross_validate(
        classifier, X_resampled, y_resampled, 
        cv=cv, scoring=scoring, return_train_score=False
    )
    classifier.fit(X_resampled, y_resampled)

    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in cv.split(X_resampled, y_resampled):
        X_train, X_test = X_resampled.iloc[train_idx], X_resampled.iloc[test_idx]
        y_train, y_test = y_resampled[train_idx], y_resampled[test_idx]

    
        if classifier_type == 'RF':
            clf = RandomForestClassifier(n_estimators=100, random_state=42) 
        elif classifier_type == 'DT':
            clf = DecisionTreeClassifier(max_depth=5, random_state=42)
        else:
            raise ValueError("Only 'RF' or 'DT' supported")
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    class_report = classification_report(
        y_true_all, y_pred_all, 
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    ) 
    
    # Feature importance
    if classifier_type == 'RF':
        importances = classifier.feature_importances_
    else:
        importances = classifier.feature_importances_
   
    results = {
        'classifier': classifier,
        'features': feature_subset,
        'cv_accuracy': round(np.mean(cv_results['test_accuracy']), 4),
        'cv_accuracy_std': round(np.std(cv_results['test_accuracy']), 4),
        'cv_mcc': round(np.mean(cv_results['test_mcc']), 4),
        'cv_mcc_std': round(np.std(cv_results['test_mcc']), 4),
        'cv_f1_macro': round(np.mean(cv_results['test_f1_macro']), 4),
        'cv_f1_macro_std': round(np.std(cv_results['test_f1_macro']), 4),
        'cv_f1_weighted': round(np.mean(cv_results['test_f1_weighted']), 4),
        'cv_f1_weighted_std': round(np.std(cv_results['test_f1_weighted']), 4),
        'cv_precision_macro': round(np.mean(cv_results['test_precision_macro']), 4),
        'cv_recall_macro': round(np.mean(cv_results['test_recall_macro']), 4),
        'feature_importances': {feature: round(imp, 4) for feature, imp in zip(feature_subset, importances)},
        'classification_report': class_report
    }
    
    print(f"{classifier_type}:")
    print(f"Accuracy: {np.mean(cv_results['test_accuracy']):.4f} (+/- {np.std(cv_results['test_accuracy']) * 2:.4f})")
    print(f"Ftrs: {len(feature_subset)}")
    
    return results

    
def result_summary_table(final_classifiers):
    model_name_labels = {
        'mrmr_rf_smote': 'mRMR + RF + SMOTE',
        'mrmr_dt_smote': 'mRMR + DT + SMOTE',
        'mcfs_rf_smote': 'MCFS + RF + SMOTE',
        'mcfs_dt_smote': 'MCFS + DT + SMOTE',
        'mrmr_rf_no_smote': 'mRMR + RF + (NO SMOTE)',
        'mrmr_dt_no_smote': 'mRMR + DT + (NO SMOTE)',
        'mcfs_rf_no_smote': 'MCFS + RF + (NO SMOTE)',
        'mcfs_dt_no_smote': 'MCFS + DT + (NO SMOTE)'
    }
    summary_data = []
    for model_name, results in final_classifiers.items():
        report = results['classification_report']
        summary_data.append({
            'Model': model_name_labels[model_name],
            'Accuracy': f"{report['accuracy']:.4f}", 
            'F1-score(weighted)': f"{report['weighted avg']['f1-score']:.4f}", 
            'MCC': results['cv_mcc']
        })
    
    summary_df = pd.DataFrame(summary_data)  
    return summary_df


def plot_classwise_performance_bar(report_dict, model_name):  
    classes = [cls for cls in report_dict.keys() if cls not in ['accuracy', 'macro avg', 'weighted avg']]
    metrics = ['precision', 'recall', 'f1-score'] 
    x = np.arange(len(classes))
    width = 0.25
    multiplier = 0 
    fig, ax = plt.subplots(figsize=(12, 5)) 
    for metric in metrics:
        values = [report_dict[cls][metric] for cls in classes]
        offset = width * multiplier
        rects = ax.bar(x + offset, values, width, label=metric, alpha=0.8)
        ax.bar_label(rects, padding=3, fmt='%.3f')
        multiplier += 1
    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title(f'Class-wise Performance Metrics - {model_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1.1) 
    plt.tight_layout()
    plt.show()

def create_venn_diagram(mrmr_features, mcfs_features, common_features): 
    mrmr_only = set(mrmr_features) - set(mcfs_features)
    mcfs_only = set(mcfs_features) - set(mrmr_features)
 
    plt.figure(figsize=(12, 5))
    
    venn_diagram = venn.venn2([set(mrmr_features), set(mcfs_features)], 
                             set_labels=(f'mRMR (Top {len(mrmr_features)} features)', 
                                        f'MCFS (Top {len(mcfs_features)} features)'))
 
    for patch in venn_diagram.patches:
        patch.set_alpha(0.4)
    
    venn_diagram.get_patch_by_id('10').set_color('skyblue')
    venn_diagram.get_patch_by_id('01').set_color('lightcoral')
    venn_diagram.get_patch_by_id('11').set_color('plum')
    
 
    
    plt.title('Common miRNA Features Identified by mRMR and MCFS Methods', 
              fontsize=16, fontweight='bold', pad=20)
     
    
    plt.tight_layout()
    plt.show()
    
    return mrmr_only, mcfs_only


def plot_feature_importance(final_classifiers): 
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 8))
    
    model_combinations = [
        ('mrmr_rf_smote', 'mRMR + Random Forest', 0),
        ('mrmr_dt_smote', 'mRMR + Decision Tree', 1),
        ('mcfs_rf_smote', 'MCFS + Random Forest', 2),
        ('mcfs_dt_smote', 'MCFS + Decision Tree', 3)
    ]
    
    
    # Get global max importance
    global_max_importance = 0
    for model_key in final_classifiers.keys():
        if model_key in final_classifiers:
            max_imp = max(final_classifiers[model_key]['classifier'].feature_importances_)
            global_max_importance = max(global_max_importance, max_imp)
    
    for model_key, title, idx in model_combinations:
        if model_key not in final_classifiers:
            continue
            
        results = final_classifiers[model_key]
        features = results['features']
        importances = results['classifier'].feature_importances_
        
        # Sort and get top 6 for single row layout
        sorted_idx = np.argsort(importances)[::-1]
        top_features = [features[i] for i in sorted_idx[:6]]
        top_importances = importances[sorted_idx][:6]
        
        # Create vertical bar plot
        x_pos = np.arange(len(top_features))
        
        bars = axes[idx].bar(x_pos, top_importances, color='steelblue', alpha=0.7,
                             linewidth=1.5)
        
        axes[idx].set_xticks(x_pos)
        axes[idx].set_xticklabels(top_features, rotation=45, ha='right', fontsize=14)
        
        axes[idx].set_ylabel('Importance Score', fontsize=16, fontweight='bold')
        axes[idx].set_title(f'{title}', fontsize=14, fontweight='bold', pad=20)
        
        # Add value labels
        for bar, importance in zip(bars, top_importances):
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                          f'{importance:.3f}', ha='center', va='bottom', 
                          fontsize=14, fontweight='bold')
         
        
        axes[idx].grid(True, axis='y', alpha=0.3)
        axes[idx].set_axisbelow(True)
        axes[idx].set_ylim(0, global_max_importance * 1.15)
     
    plt.tight_layout()
    plt.show()


def compare_models_classification(final_classifiers): 
    models = list(final_classifiers.keys())
    classes = [cls for cls in final_classifiers[models[0]]['classification_report'].keys() 
               if cls not in ['accuracy', 'macro avg', 'weighted avg']] 
    comparison_data = []
    for model in models:
        report = final_classifiers[model]['classification_report']
        for cls in classes:
            comparison_data.append({
                'Model': model,
                'Class': cls,
                'Precision': report[cls]['precision'],
                'Recall': report[cls]['recall'],
                'F1-Score': report[cls]['f1-score']
            })
    
    comparison_df = pd.DataFrame(comparison_data) 
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for idx, metric in enumerate(['Precision', 'Recall', 'F1-Score']):
        pivot_data = comparison_df.pivot(index='Class', columns='Model', values=metric)
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                   center=0.8, vmin=0, vmax=1, ax=axes[idx])
        axes[idx].set_title(f'{metric} Comparison', fontweight='bold')
    
    plt.suptitle('Multi-Model Classification Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return comparison_df
