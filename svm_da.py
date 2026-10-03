from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

def svm_da(final_brain_df, brain_cols, df_hr, ef_col, perform_norm_modeling):

    # ---------------------------------------------------------
    # Prepare data
    # ---------------------------------------------------------
    if perform_norm_modeling:
        brain_cols = [col + '_z' for col in brain_cols]

    df_ef = final_brain_df[['Identifiers', ef_col, 'Group']].copy()

    df_all = pd.merge(
        df_ef,
        df_hr,
        on="Identifiers",
        how="inner"
    )

    df_all = df_all.dropna().reset_index(drop=True)

    X_brain = df_all[brain_cols].copy()
    y_EF = df_all[ef_col].copy()

    # ---------------------------------------------------------
    # Select extreme EF groups
    # ---------------------------------------------------------
    q_low = y_EF.quantile(0.25)
    q_high = y_EF.quantile(0.75)

    mask = (y_EF < q_low) | (y_EF > q_high)

    X_group = X_brain[mask].reset_index(drop=True)
    y_group = y_EF[mask].reset_index(drop=True)

    # 0 = Low EF, 1 = High EF
    y_group = (y_group > q_high).astype(int)

    print(
        f"SVM subjects in extreme groups: {len(y_group)} "
        f"(Low EF: {(y_group == 0).sum()}, "
        f"High EF: {(y_group == 1).sum()})"
    )
    plot_svm_boundary_2d(X_group, y_group)
    # ---------------------------------------------------------
    # Same nested CV structure as PLS
    # ---------------------------------------------------------
    outer_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    inner_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=43
    )

    # ---------------------------------------------------------
    # SVM pipeline
    #
    # Scaling happens separately within each CV fold.
    # ---------------------------------------------------------
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='linear'))
    ])

    # Tune the SVM regularization parameter
    param_grid = {
        'svm__C': [0.001, 0.01, 0.1, 1, 10, 100]
    }

    # ---------------------------------------------------------
    # Nested CV
    # ---------------------------------------------------------
    fold_aucs = []

    for train_idx, test_idx in outer_cv.split(X_group, y_group):

        X_train = X_group.iloc[train_idx]
        X_test = X_group.iloc[test_idx]

        y_train = y_group.iloc[train_idx]
        y_test = y_group.iloc[test_idx]

        grid = GridSearchCV(
            pipeline,
            param_grid,
            scoring='roc_auc',
            cv=inner_cv,
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        # Continuous SVM decision score
        y_test_score = grid.decision_function(X_test)

        fold_auc = roc_auc_score(
            y_test,
            y_test_score
        )

        fold_aucs.append(fold_auc)

    mean_auc = np.mean(fold_aucs)

    print(f"{ef_col} SVM nested CV AUC: {mean_auc:.3f}")

    # ---------------------------------------------------------
    # Permutation test
    # ---------------------------------------------------------
    n_permutations = 1000

    def svm_permutation(seed):

        rng = np.random.RandomState(seed)

        y_perm = pd.Series(
            rng.permutation(y_group.values)
        )

        perm_fold_aucs = []

        for train_idx, test_idx in outer_cv.split(
            X_group, y_perm
        ):

            X_train = X_group.iloc[train_idx]
            X_test = X_group.iloc[test_idx]

            y_train = y_perm.iloc[train_idx]
            y_test = y_perm.iloc[test_idx]

            grid = GridSearchCV(
                pipeline,
                param_grid,
                scoring='roc_auc',
                cv=inner_cv,
                n_jobs=-1
            )

            grid.fit(X_train, y_train)

            y_test_score = grid.decision_function(X_test)

            perm_fold_aucs.append(
                roc_auc_score(
                    y_test,
                    y_test_score
                )
            )

        return np.mean(perm_fold_aucs)

    perm_aucs = Parallel(n_jobs=-1)(
        delayed(svm_permutation)(seed)
        for seed in tqdm(
            range(n_permutations),
            desc="SVM permutation test"
        )
    )

    perm_aucs = np.array(perm_aucs)

    p_value = (
        np.sum(perm_aucs >= mean_auc) + 1
    ) / (
        n_permutations + 1
    )

    print(f"{ef_col} SVM permutation p-value: {p_value:.3f}")

    # ---------------------------------------------------------
    # Final SVM feature weights
    # ---------------------------------------------------------
    final_grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring='roc_auc',
        cv=inner_cv,
        n_jobs=-1
    )

    final_grid.fit(X_group, y_group)

    best_svm = final_grid.best_estimator_.named_steps['svm']
    weights = best_svm.coef_[0]

    importance_df = pd.DataFrame({
        'Region': brain_cols,
        'Weight': weights,
        'Abs_Weight': np.abs(weights)
    }).sort_values(
        by='Abs_Weight',
        ascending=False
    )

    print(f"\n{ef_col} Top 10 SVM feature weights:")
    print(importance_df.head(10).to_string(index=False))

    return mean_auc, p_value

def plot_svm_boundary_2d(X, y):
    """
    Fit a linear SVM and visualize its classification boundary
    in a 2-D space defined by the two most important SVM directions.
    """

    # ---------------------------------------------------------
    # Standardize the brain measures
    # ---------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------------------------------------------------------
    # Fit linear SVM
    # ---------------------------------------------------------
    svm = SVC(kernel='linear', C=1)
    svm.fit(X_scaled, y)

    # ---------------------------------------------------------
    # SVM weight vector
    # ---------------------------------------------------------
    w = svm.coef_[0]

    # First direction = SVM weight vector
    direction1 = w / np.linalg.norm(w)

    # Find a second direction orthogonal to direction1.
    # We use PCA on the data after removing the SVM direction.
    X_remaining = X_scaled - np.outer(
        X_scaled @ direction1,
        direction1
    )

    _, _, vh = np.linalg.svd(X_remaining, full_matrices=False)

    direction2 = vh[0]

    # Make sure direction2 is orthogonal to direction1
    direction2 = direction2 - (
        direction2 @ direction1
    ) * direction1
    direction2 = direction2 / np.linalg.norm(direction2)

    # ---------------------------------------------------------
    # Project subjects into 2-D
    # ---------------------------------------------------------
    x1 = X_scaled @ direction1
    x2 = X_scaled @ direction2

    # ---------------------------------------------------------
    # Calculate SVM boundary in this 2-D representation
    # ---------------------------------------------------------
    # Since direction1 is the SVM weight direction,
    # the boundary is approximately vertical in this projection.
    boundary_x1 = -svm.intercept_[0] / np.linalg.norm(w)

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 7))

    low = y.to_numpy() == 0
    high = y.to_numpy() == 1

    plt.scatter(
        x1[low],
        x2[low],
        label='Low EF',
        alpha=0.75
    )

    plt.scatter(
        x1[high],
        x2[high],
        label='High EF',
        alpha=0.75
    )

    plt.axvline(
        boundary_x1,
        linestyle='--',
        linewidth=2,
        label='SVM boundary'
    )

    plt.xlabel('SVM direction')
    plt.ylabel('Orthogonal brain-data direction')
    plt.title('Linear SVM classification of extreme EF groups')
    plt.legend()
    plt.tight_layout()
    plt.show()