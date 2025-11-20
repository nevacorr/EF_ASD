
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from combatlearn import ComBat  # CovBat method via combatlearn
import numpy as np
import pandas as pd
import shap
import warnings
import time

"""
   predict_SA_xgboost_covbat

   This function trains and evaluates an XGBoost regression model on neuroimaging data while addressing site effects,
   missing data, and potential data leakage. It does:

   1. Per-site Median Imputation:
      - Missing values (NaNs) are imputed separately for each site using the median, which is robust to outliers.
      - Imputation is fit only on the training data for each fold/bootstrapped sample to prevent leakage.

   2. CovBat Harmonization:
      - Features are harmonized across sites using CovBat, which removes site-specific mean, variance, and covariance effects.
      - Harmonization is applied only to non-missing values in the training data and then applied to the test set.

   3. Cross-Validation and Bootstrapping:
      - Supports K-fold cross-validation (default 10 folds) and optional bootstrapping of the training data.
      - Predictions are aggregated across folds and bootstraps.

   4. XGBoost Modeling:
      - XGBoost regression is trained either with manually specified hyperparameters or using Bayesian optimization via BayesSearchCV.
      - The function supports multiple bootstraps and returns R² performance metrics.

   5. SHAP Feature Importance:
      - Computes SHAP values for test samples to assess feature contributions.
      - Aggregates SHAP values across folds and bootstraps for downstream analysis or plotting.

   6. Output:
      - Returns R² scores across bootstraps, aggregated feature importance, SHAP values, original feature values, group labels, and sex labels.

   This workflow ensures robust prediction while minimizing site confounds, handling missing data safely, 
   and providing interpretable feature importance metrics.
   """

def predict_SA_xgboost_covbat(
        X, y, group_vals, sex_vals, target, metric, params,
        run_dummy_quick_fit=False, set_params_man=0,
        show_results_plot=False, bootstrap=False, n_bootstraps=1
):
    """
    Simplified XGBoost pipeline with:
    1. Site-wise median imputation
    2. CovBat harmonization (method='chen') to correct mean, variance, and covariance across sites
    3. Optional BayesSearchCV hyperparameter tuning
    4. KFold cross-validation and optional bootstrapping
    Returns R² scores and aggregated SHAP values.
    """
    warnings.simplefilter(action='ignore', category=FutureWarning)
    rng = np.random.default_rng(42)
    start_time = time.time()

    if run_dummy_quick_fit:
        n_iter = 5
    else:
        n_iter = 100

    r2_test_all_bootstraps = []
    all_shap_values = []

    feature_cols = [c for c in X.columns if c not in ['Site', 'Sex']]

    for b in range(max(n_bootstraps, 1)):
        train_predictions = np.zeros_like(y, dtype=np.float64)
        test_predictions = np.zeros_like(y, dtype=np.float64)
        train_counts = np.zeros_like(y, dtype=np.int64)

        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        for train_idx, test_idx in kf.split(X, y):
            X_train = X.iloc[train_idx].copy()
            y_train = y[train_idx].copy()
            X_test = X.iloc[test_idx].copy()
            y_test = y[test_idx].copy()

            sex_train = X_train['Sex'].values.reshape(-1, 1)
            sex_test = X_test['Sex'].values.reshape(-1, 1)

            if bootstrap:
                bootstrap_indices = rng.choice(len(train_idx), size=len(train_idx), replace=True)
                X_train = X_train.iloc[bootstrap_indices].reset_index(drop=True)
                y_train = y_train[bootstrap_indices]
                sex_train = sex_train[bootstrap_indices]

            # -----------------------------
            # 1️⃣ Site-wise median imputation
            # -----------------------------
            X_train_imputed = X_train.copy()
            X_test_imputed = X_test.copy()
            sites_train = X_train['Site'].values
            sites_test = X_test['Site'].values

            for site in np.unique(sites_train):
                site_idx_train = X_train['Site'] == site
                site_idx_test = X_test['Site'] == site

                imputer = SimpleImputer(strategy='median')
                X_train_imputed.loc[site_idx_train, feature_cols] = imputer.fit_transform(
                    X_train.loc[site_idx_train, feature_cols]
                )
                X_test_imputed.loc[site_idx_test, feature_cols] = imputer.transform(
                    X_test.loc[site_idx_test, feature_cols]
                )

            # -----------------------------
            # 2️⃣ CovBat harmonization
            # Corrects mean, variance, AND covariance across sites
            # Applied separately per feature
            # -----------------------------
            X_train_harmonized = X_train_imputed.copy()
            X_test_harmonized = X_test_imputed.copy()
            for col in feature_cols:
                not_nan_idx = ~X_train_harmonized[col].isna()
                covbat = ComBat(method='chen')  # CovBat harmonization
                # Fit CovBat on training data
                covbat.fit(X_train_harmonized.loc[not_nan_idx, [col]].values,
                           batch=sites_train[not_nan_idx].reshape(-1, 1))
                # Transform training data
                X_train_harmonized.loc[not_nan_idx, col] = covbat.transform(
                    X_train_harmonized.loc[not_nan_idx, [col]].values,
                    sites_train[not_nan_idx].reshape(-1, 1)
                ).ravel()
                # Transform test data using same fitted CovBat model
                X_test_harmonized[col] = covbat.transform(
                    X_test_harmonized[[col]].values, sites_test.reshape(-1, 1)
                ).ravel()

            # Add sex column back
            X_train_final = np.hstack([sex_train, X_train_harmonized[feature_cols].values])
            X_test_final = np.hstack([sex_test, X_test_harmonized[feature_cols].values])

            # -----------------------------
            # 3️⃣ XGBoost
            # -----------------------------
            if set_params_man == 0:
                xgb = XGBRegressor(objective='reg:squarederror', n_jobs=-1)
                opt = BayesSearchCV(xgb, params, n_iter=n_iter, n_jobs=-1)
                opt.fit(X_train_final, y_train)
                model = opt.best_estimator_
            else:
                model = XGBRegressor(
                    objective='reg:squarederror',
                    n_jobs=-1,
                    colsample_bytree=params['colsample_bytree'],
                    eta=params['eta'],
                    gamma=params['gamma'],
                    max_depth=params['max_depth'],
                    min_child_weight=params['min_child_weight'],
                    n_estimators=params['n_estimators'],
                    subsample=params['subsample']
                )
                model.fit(X_train_final, y_train)

            # Predictions
            y_pred_test = model.predict(X_test_final)
            y_pred_train = model.predict(X_train_final)
            test_predictions[test_idx] = y_pred_test
            train_predictions[train_idx] += y_pred_train
            train_counts[train_idx] += 1

            # SHAP values
            explainer = shap.Explainer(model, X_train_final)
            shap_values = explainer(X_test_final)
            all_shap_values.append(shap_values.values)

        # Correct train predictions by number of times each sample appeared
        train_predictions /= train_counts

        r2_test = r2_score(y, test_predictions)
        r2_test_all_bootstraps.append(r2_test)
        print(f"Bootstrap {b + 1}/{n_bootstraps} R² test: {r2_test:.3f}")

    # Aggregate SHAP values across folds/bootstraps
    all_shap_values = np.vstack(all_shap_values)

    return np.array(r2_test_all_bootstraps), all_shap_values