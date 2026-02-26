import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

def pls_da(final_brain_df, brain_cols, df_hr, ef_col):
    # -----------------------------
    # Step 1: Prepare your data
    # -----------------------------
    # X_brain: subjects x brain features (numpy array or DataFrame)
    # y_EF: continuous EF scores (numpy array or Series)

    # Example placeholder data
    # X_brain = pd.DataFrame(np.random.rand(50, 100))  # 50 subjects, 100 brain features
    # y_EF = pd.Series(np.random.rand(50))

    df_ef=final_brain_df[['Identifiers', ef_col]].copy()
    df_all=pd.merge(df_ef, df_hr,  on="Identifiers", how="inner")
    df_all=df_all.dropna().reset_index(drop=True)

    X_brain = df_all[brain_cols].copy()
    y_EF = df_all[ef_col].copy()

    # -----------------------------
    # Step 2: Extreme group selection
    # -----------------------------
    q_low = y_EF.quantile(0.25)
    q_high = y_EF.quantile(0.75)
    mask = (y_EF <= q_low) | (y_EF >= q_high)

    X_group = X_brain[mask]
    y_group = y_EF[mask].copy()
    cov_group = covariates[mask]

    # Binary labels: 0 = Low EF, 1 = High EF
    y_group[:] = (y_group > q_high).astype(int)

    # -----------------------------
    # Step 3: Covariate adjustment (residualization)
    # -----------------------------
    X_resid = pd.DataFrame(index=X_group.index, columns=X_group.columns)

    for col in X_group.columns:
        X_col = X_group[col]
        model = sm.OLS(X_col, sm.add_constant(cov_group)).fit()
        X_resid[col] = model.resid

    # -----------------------------
    # Step 4: Standardize features
    # -----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resid)

    # -----------------------------
    # Step 5: Train/Test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_group, test_size=0.2, stratify=y_group, random_state=42
    )

    # -----------------------------
    # Step 6: Cross-validation to select n_components
    # -----------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Maximum allowed components = min(n_samples_train, n_features)
    max_components = min(X_train.shape[0], X_train.shape[1], 5)  # optional cap at 5

    best_auc = 0
    best_n = 1

    for n_comp in range(1, max_components + 1):
        pls = PLSRegression(n_components=n_comp)
        aucs = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            pls.fit(X_train[train_idx], y_train.iloc[train_idx])
            y_val_pred = pls.predict(X_train[val_idx]).ravel()
            aucs.append(roc_auc_score(y_train.iloc[val_idx], y_val_pred))
        mean_auc = np.mean(aucs)
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_n = n_comp

    print(f"Selected n_components: {best_n}, CV AUC: {best_auc:.3f}")

    # -----------------------------
    # Step 7: Train final model on full training set
    # -----------------------------
    pls_final = PLSRegression(n_components=best_n)
    pls_final.fit(X_train, y_train)

    # Evaluate on test set
    y_test_pred = pls_final.predict(X_test).ravel()
    test_auc = roc_auc_score(y_test, y_test_pred)
    print(f"Test ROC AUC: {test_auc:.3f}")

    # -----------------------------
    # Step 8: Feature importance
    # -----------------------------
    feature_weights = pls_final.x_weights_[:, 0]  # first PLS component
    brain_feature_importance = pd.Series(feature_weights, index=X_brain.columns).sort_values(ascending=False)

    print("Top 10 features contributing to Low vs High EF separation:")
    print(brain_feature_importance.head(10))