from skopt import BayesSearchCV
from xgboost import XGBRegressor
import numpy as np
import time
from sklearn.metrics import r2_score
import warnings
from Utility_Functions_XGBoost import write_modeling_data_and_outcome_to_file, aggregate_feature_importances
from covbat_harmonize import covbat_harmonize
from skopt.callbacks import VerboseCallback
from sklearn.model_selection import train_test_split

def predict_SA_xgboost_covbat(
        X, y, target, metric, params, run_dummy_quick_fit,
        n_repeats=20, test_size=0.2, random_state=42,  X_test=None, y_test=None):

    """
    Train XGBoost with hyperparameter tuning and evaluate:
      - repeated train/validation splits on main data
      - optional completely unseen test set

    Returns:
        r2_vals: array of validation R² for each repeat
        feature_importance_df: aggregated feature importances
        r2_test (optional): R² on unseen test set if X_test/y_test provided
    """

    # Suppress all FutureWarnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # set number of iterations for BayesCV
    n_iter = 5 if run_dummy_quick_fit else 50  #can increase

    r2_vals = []
    feature_importance_list = []
    best_params_list = []

    rng =  np.random.default_rng(random_state)

    # record start time
    start_time = time.time()

    for rep in range(n_repeats):

        print(f"\nRepeat {rep +1}/{n_repeats}")

        #----------------------------------
        #  Train/ validation split
        #----------------------------------
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=rng.integers(0, 1e6))

        # ----------------------------------
        #  CovBat harmonization
        # ----------------------------------
        X_train_h, X_val_h = covbat_harmonize(X_train, X_val)

        # ----------------------------------
        # Hyperparameter search (TRAIN only)
        # ----------------------------------
        xgb = XGBRegressor(objective="reg:squarederror", n_jobs=-1)
        opt = BayesSearchCV(xgb, params, n_iter=n_iter, cv=3, n_jobs=-1, verbose=0)

        opt.fit(X_train_h, y_train, callback=[VerboseCallback(n_total=n_iter)])
        best_model = opt.best_estimator_
        best_params_list.append(opt.best_params_)

        #----------------------------------
        # Validation performance
        #----------------------------------
        val_preds = best_model.predict(X_val_h)
        r2_val = r2_score(y_val, val_preds)
        print(f"R2 (validation) = {r2_val:.3f}")
        r2_vals.append(r2_val)

        # Store feagure importances
        feature_importance_list.append(best_model.feature_importances_)

    # ----------------------------------
    # Aggregate results
    # ----------------------------------
    r2_vals = np.array(r2_vals)

    feature_names = ['Sex'] + X.drop(columns=['Site', 'Sex']).columns.tolist()
    feature_importance_df = aggregate_feature_importances(
        feature_importance_list, feature_names, n_repeats,
        outputfilename=f"{target}_{metric}_{n_repeats}_xgb_feature_importance.txt",
        top_n=25)
    # ----------------------------------
    # Optional test set evaluation
    # ----------------------------------
    r2_test = None
    if X_test is not None and y_test is not None:
        final_params = {}
        # Choose hyperparameter combination that had the best validation R2
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


    return r2_vals, feature_importance_df, r2_test
