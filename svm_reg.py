import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

def svr_reg(final_brain_df, brain_cols, df_hr, ef_col, perform_norm_modeling):

    # ---------------- Prepare data (same as svm_da) ----------------
    if perform_norm_modeling:
        brain_cols = [col + '_z' for col in brain_cols]

    df_ef = final_brain_df[['Identifiers', ef_col, 'Group']].copy()
    df_all = pd.merge(df_ef, df_hr, on="Identifiers", how="inner")
    df_all = df_all.dropna().reset_index(drop=True)

    X = df_all[brain_cols].copy()
    y = df_all[ef_col].copy()

    # Standardize EF so epsilon is on a sensible scale (2 numbers, negligible leakage)
    y = (y - y.mean()) / y.std()

    print(f"SVR subjects: {len(y)}")

    # ---------------- Nested CV ----------------
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=43)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf'))
    ])

    param_grid = {
        'svr__C': [0.01, 0.1, 1, 10, 100],
        'svr__gamma': ['scale', 0.001, 0.01, 0.1],
        'svr__epsilon': [0.05, 0.1, 0.2]
    }

    fold_rho, fold_r2 = [], []
    all_true, all_pred = [], []

    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        grid = GridSearchCV(pipeline, param_grid,
                            scoring='neg_mean_squared_error',
                            cv=inner_cv, n_jobs=-1)
        grid.fit(X_train, y_train)

        y_pred = grid.predict(X_test)

        fold_rho.append(spearmanr(y_test, y_pred)[0])
        fold_r2.append(r2_score(y_test, y_pred))
        all_true.extend(y_test)
        all_pred.extend(y_pred)

    mean_rho = np.mean(fold_rho)
    print(f"{ef_col} SVR Spearman rho: {mean_rho:.3f} (SD {np.std(fold_rho):.3f})")
    print(f"{ef_col} SVR R2: {np.mean(fold_r2):.3f}  (0 = no better than predicting the mean)")

    # Predicted vs actual
    plt.figure(figsize=(6, 6))
    plt.scatter(all_true, all_pred, alpha=0.5)
    plt.xlabel('Actual EF (standardized)')
    plt.ylabel('Predicted EF (out-of-fold)')
    plt.title(f'SVR: rho = {mean_rho:.2f}')
    plt.tight_layout()
    plt.show()

    return mean_rho, np.mean(fold_r2), df_all