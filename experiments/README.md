# Experiments

This folder contains the Python experiment pipeline for running the current
baseline study in a cleaner way.

The first pipeline keeps the current study design:

1. use the saved Boruta-filtered expression matrix,
2. use the saved mRMR and MCFS rankings,
3. run incremental feature selection with Random Forest and Decision Tree,
4. evaluate the final feature subsets with and without SMOTE.

The important change is that SMOTE is applied only inside each training fold.
The validation fold is always left untouched.

Run from the repository root:

```bash
python -m experiments.run_baseline_experiment
```

You can also run it directly if your IDE launches files as scripts:

```bash
python experiments/run_baseline_experiment.py
```

Outputs are written to `experiments/output/`.

Figures are written to `experiments/output/plots/`.
