import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from plot_pls_scores import plot_pls_scores

def run_cv_pipeline(X, y, max_components, outer_cv, inner_cv):
    """
    Runs nested cross-validation:
      - Outer loop: evaluates model AUC on held-out fold
      - Inner loop: selects optimal number of PLS components
    Standardization is performed inside each outer fold to prevent leakage.
    Note: PLSRegression is used with binary Y (0/1) to perform PLS-DA —
    sklearn has no separate PLS-DA class.
    Returns mean AUC across outer folds.
    """
    fold_aucs = []
    pls1_scores = []
    pls1_labels = []

    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # ── Standardize inside fold (fit on train only) ───────────────────────
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        # ── Inner CV: select optimal n_components ─────────────────────────────
        best_inner_auc = -np.inf
        best_n = 1
        for n_comp in range(1, max_components + 1):
            pls = PLSRegression(n_components=n_comp)
            inner_aucs = []
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train, y_train):
                # Scale within each inner fold
                scaler_inner = StandardScaler()
                X_inner_train = scaler_inner.fit_transform(X_train.iloc[inner_train_idx])
                X_inner_val = scaler_inner.transform(X_train.iloc[inner_val_idx])
                pls.fit(X_inner_train, y_train.iloc[inner_train_idx])
                y_val_pred = pls.predict(X_inner_val).ravel()
                inner_aucs.append(roc_auc_score(y_train.iloc[inner_val_idx], y_val_pred))
            mean_inner_auc = np.mean(inner_aucs)
            if mean_inner_auc > best_inner_auc:
                best_inner_auc = mean_inner_auc
                best_n = n_comp

        # ── Fit on full outer fold train set, evaluate on held-out test ───────
        pls_fold = PLSRegression(n_components=best_n)
        pls_fold.fit(X_train_scaled, y_train)
        y_test_pred = pls_fold.predict(X_test_scaled).ravel()
        fold_aucs.append(roc_auc_score(y_test, y_test_pred))

        # Out-of-fold PLS1 scores for visualization
        y_test_scores = pls_fold.transform(X_test_scaled)[:, 0]

        pls1_scores.extend(y_test_scores)
        pls1_labels.extend(y_test)

    return np.mean(fold_aucs), np.array(pls1_scores), np.array(pls1_labels)


def single_permutation(X, y, max_components):
    """
    Runs one permutation — shuffles labels and reruns the full nested CV pipeline
    with fresh CV splits (random_state=None) so each permutation uses different folds.
    """
    y_perm   = pd.Series(np.random.permutation(y), index=y.index)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)

    perm_auc, _, _ = run_cv_pipeline(X, y_perm, max_components, outer_cv, inner_cv)

    return perm_auc


def pls_da(final_brain_df, brain_cols, df_hr, ef_col, perform_norm_modeling):

    # Prepare data ──────────────────────────────────────────────────
    if perform_norm_modeling:
        brain_cols = [col + '_z' for col in brain_cols]

    df_ef  = final_brain_df[['Identifiers', ef_col, 'Group']].copy()
    df_all = pd.merge(df_ef, df_hr, on="Identifiers", how="inner")
    df_all = df_all.dropna().reset_index(drop=True)

    X_brain = df_all[brain_cols].copy()
    y_EF    = df_all[ef_col].copy()
    X_Group = df_all['Group'].copy()

    # ── Step 2: Extreme group selection ──────────────────────────────────────
    q_low  = y_EF.quantile(0.25)
    q_high = y_EF.quantile(0.75)
    mask   = (y_EF < q_low) | (y_EF > q_high)

    X_group = X_brain[mask].reset_index(drop=True)
    y_group = y_EF[mask].copy().reset_index(drop=True)
    y_group[:] = (y_group > q_high).astype(int)   # 0 = Low EF, 1 = High EF

    print(f"Subjects in extreme groups: {len(y_group)} "
          f"(Low EF: {(y_group==0).sum()}, High EF: {(y_group==1).sum()})")

    max_components = min(X_group.shape[0] // 2, X_group.shape[1], 5)

    # Different seeds for outer and inner CV to prevent correlated splits
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)

    # Nested CV — observed AUC ─────────────────────────────────────
    mean_auc, pls1_scores, pls1_labels = run_cv_pipeline(X_group, y_group, max_components, outer_cv, inner_cv)
    print(f"Nested CV AUC: {mean_auc:.3f}")

    # Permutation test — parallel, each permutation uses fresh CV splits ──
    n_permutations = 1000
    perm_aucs = Parallel(n_jobs=-1)(
        delayed(single_permutation)(X_group, y_group, max_components)
        for _ in tqdm(range(n_permutations), desc="Permutation test")
    )

    perm_aucs = np.array(perm_aucs)
    p_value   = (np.sum(perm_aucs >= mean_auc) + 1) / (n_permutations + 1)
    print(f"Permutation p-value: {p_value:.3f}")

    # Fit final model on ALL data for feature importance ────────────

    # Select best n_components using full-data inner CV
    best_auc_final = -np.inf
    best_n_final   = 1
    for n_comp in range(1, max_components + 1):
        pls = PLSRegression(n_components=n_comp)
        fold_aucs = []
        for train_idx, val_idx in inner_cv.split(X_group, y_group):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_group.iloc[train_idx])
            X_val = scaler.transform(X_group.iloc[val_idx])
            pls.fit(X_train, y_group.iloc[train_idx])
            y_val_pred = pls.predict(X_val).ravel()
            fold_aucs.append(roc_auc_score(y_group.iloc[val_idx], y_val_pred))
        mean_fold_auc = np.mean(fold_aucs)
        if mean_fold_auc > best_auc_final:
            best_auc_final = mean_fold_auc
            best_n_final = n_comp
    scaler_final = StandardScaler()
    X_all_scaled = scaler_final.fit_transform(X_group)
    pls_final = PLSRegression(n_components=best_n_final)
    pls_final.fit(X_all_scaled, y_group)
    print(f"Final model n_components: {best_n_final}")
    # Plot first two components
    plot_pls_scores(pls_final, y_group)

    # Feature importance ────────────────────────────────────────────
    # Feature weights for all PLS components
    feature_weights = pd.DataFrame(
        pls_final.x_weights_,
        index=brain_cols,
        columns=[f"Component {i + 1}" for i in range(pls_final.x_weights_.shape[1])]
    )

    for component in feature_weights.columns:
        print(f"\nTop features contributing to {component}:")

        component_weights = feature_weights[component]
        importance = component_weights.reindex(
            component_weights.abs().sort_values(ascending=False).index
        )

        print(importance.head(10))

    return mean_auc, p_value, importance, X_Group
