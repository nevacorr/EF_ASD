from skopt import BayesSearchCV
from xgboost import XGBRegressor
import numpy as np
import time
from sklearn.metrics import r2_score
import warnings
from Utility_Functions_XGBoost import write_modeling_data_and_outcome_to_file, aggregate_feature_importances
from covbat_harmonize import covbat_harmonize
from skopt.callbacks import VerboseCallback
from sklearn.model_selection import KFold
import json
import os
from sklearn.utils import resample
from analyze_hyperparameter_repeats import analyze_hyperparameter_repeats


def predict_SA_xgboost_nm(
        X, y, target, metric, params, run_dummy_quick_fit,
        n_outer_splits=10, random_state=42,  X_test=None, y_test=None):

    """
    Nested CV with BayesSearchCV (inner loop) and bootstrapping (outer CI)
    - Inner loop: hyperparameter tuning on outer training fold
    - Outer loop: Evaluate R² on held-out outer fold
    - Bootstrapping: Resample data to compute CI for mean R²
    """
    working_dir = os.getcwd()

    # Suppress all FutureWarnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # set number of iterations for BayesCV
    n_iter = 5 if run_dummy_quick_fit else 50

    r2_vals = []
    feature_importance_list = []
    best_params_list = []

    rng =  np.random.default_rng(random_state)

    # record start time
    start_time = time.time()

    #Outer loop: KFold over the main dataset
    outer_cv = KFold(n_splits=n_outer_splits, shuffle=True, random_state=random_state)
    for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X)):

        print(f"\nOuter Fold {fold_idx+1}/{n_outer_splits}")

        #----------------------------------
        #  Train/ validation split
        #----------------------------------
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # ----------------------------------
        # Inner loop: hyperparameter tuning on outer training fold
        # ----------------------------------
        xgb = XGBRegressor(verbosity=0, objective="reg:squarederror", n_jobs=-1)
        opt = BayesSearchCV(xgb, params, n_iter=n_iter, cv=3, n_jobs=-1, verbose=0)
        opt.fit(X_train, y_train, callback=[VerboseCallback(n_total=n_iter)])
        best_model = opt.best_estimator_
        best_params_list.append(opt.best_params_)

        #----------------------------------
        # Evaluate R2 on outer validation fold
        #----------------------------------
        val_preds = best_model.predict(X_val)
        r2_val = r2_score(y_val, val_preds)
        print(f"R2 (validation fold) = {r2_val:.3f}")
        r2_vals.append(r2_val)

        # Store feagure importances
        feature_importance_list.append(best_model.feature_importances_)

    # ----------------------------------
    # Aggregate results
    # ----------------------------------
    r2_vals = np.array(r2_vals)

    feature_names = ['Sex'] + X.drop(columns=['Site', 'Sex']).columns.tolist()
    feature_importance_df = aggregate_feature_importances(
        feature_importance_list, feature_names, n_outer_splits,
        outputfilename=f"{working_dir}/{target}_{metric}_n_outer_splits_{n_outer_splits}_xgb_feature_importance.txt",
        top_n=25)
    # ----------------------------------
    # Optional test set evaluation
    # ----------------------------------
    r2_test = None
    if X_test is not None and y_test is not None:
        # Use hyperparameters from outer fold with best R2
        best_idx = np.argmax(r2_vals)
        final_params = best_params_list[best_idx]

        # Fit final model on entire main dataset
        X_full_h, X_test_h = covbat_harmonize(X, X_test)
        final_model = XGBRegressor(objective="reg:squarederror", n_jobs=-1, **final_params)
        final_model.fit(X_full_h, y)
        test_preds = final_model.predict(X_test_h)
        r2_test = r2_score(y_test, test_preds)
        print(f"R2 on test set: {r2_test:.3f}")

    elapsed_time = (time.time() - start_time) / 60.0
    print(f"\nAll repeats complete. Time since beginning of program: {elapsed_time:.2f} minutes")

    # if bootstrap == 0:
    #     write_modeling_data_and_outcome_to_file(run_dummy_quick_fit, metric, params, set_parameters_manually, target, X,
    #                                          r2_train, r2_val, best_params, bootstrap, elapsed_time)

    # Save results
    with open(f"{working_dir}/{target}_{metric}_best_params.txt", "w") as f:
        json.dump(best_params_list, f, indent=4)

    np.save(f"{working_dir}/{target}_{metric}_{n_iter}_{n_outer_splits}_r2_vals.npy", np.array(r2_vals))

    # Confidence interval
    ci_percentile = 95
    lower = np.percentile(r2_vals, (100 - ci_percentile) / 2)
    upper = np.percentile(r2_vals, 100 - (100 - ci_percentile) / 2)
    print(f"{ci_percentile:.1f}% confidence interval: [{lower:.4f}, {upper:.4f}]")

    ci_file = f"{working_dir}/{target}_{metric}_{n_iter}_n_outer_splits_{n_outer_splits}_r2_summary.txt"
    with open(ci_file, "w") as f:
        f.write(f"Validation R2 (all folds): {r2_vals.tolist()}\n")
        f.write(f"Mean validation R2: {np.mean(r2_vals):.4f}\n")
        f.write(f"Best validation R2: {np.max(r2_vals):.4f}\n")
        f.write(f"{ci_percentile}% confidence interval: [{lower:.4f}, {upper:.4f}]\n")
        if r2_test is not None:
            f.write(f"R2 on unseen test set: {r2_test:.4f}\n")

    analyze_hyperparameter_repeats(best_params_list, r2_vals, top_n=5)

    return r2_vals, feature_importance_df, r2_test
