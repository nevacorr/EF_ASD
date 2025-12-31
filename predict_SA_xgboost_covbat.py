from skopt import BayesSearchCV
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import numpy as np
import time
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import shap
from Utility_Functions_XGBoost import write_modeling_data_and_outcome_to_file, aggregate_feature_importances
from Utility_Functions_XGBoost import plot_top_shap_scatter_by_group, plot_top_shap_distributions_by_group
from Utility_Functions_XGBoost import plot_shap_magnitude_histograms_equal_bins, plot_shap_magnitude_by_sex_and_group
from Utility_Functions_XGBoost import plot_shap_magnitude_kde
from covbat_harmonize import covbat_harmonize

def predict_SA_xgboost_covbat(X, y, group_vals, sex_vals, target, metric, params, run_dummy_quick_fit, set_params_man,
                       show_results_plot, bootstrap, n_bootstraps, X_test, y_test, include_asd_in_train):

    # Suppress all FutureWarnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    set_parameters_manually = set_params_man

    r2_val_all_bootstraps=[]

    # set number of iterations for BayesCV
    if run_dummy_quick_fit:
        n_iter = 5
    else:
        n_iter = 100

    if set_parameters_manually == 0: #if search for best parameters

        xgb = XGBRegressor(objective="reg:squarederror", n_jobs=-1)
        opt = BayesSearchCV(xgb, params, n_iter=n_iter, n_jobs=-1)

    else:  # if parameters are to be set manually at fixed values

        xgb = XGBRegressor(
            objective="reg:squarederror",
            n_jobs=16,
            colsample_bytree=params["colsample_bytree"],#the fraction of features to be selected for each tree
            eta=params["eta"],  # learning rate
            gamma=params['gamma'],  # regularization. Low values allow splits as long as they improve the loss function, no matter how small
            max_depth=params["max_depth"],#maximum depth of each decision tree
            min_child_weight=params["min_child_weight"],# the number of samples required in each child node before attempting to split further
            n_estimators=params["n_estimators"],  # Number of trees to create during training
            subsample=params["subsample"]  # Fraction of training dta that is sampled for each boosting round
        )
    # record start time
    start_time = time.time()
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility

    if bootstrap == 0:
        n_bootstraps = 1 # force one iteration with no bootstrapping

    feature_importance_list = []

    for b in range(n_bootstraps):

        # make variables to hold predictions for train and validation run, as well as counts for how many times each subject appears in a train set
        train_predictions = np.zeros_like(y, dtype=np.float64)
        val_predictions = np.zeros_like(y, dtype=np.float64)
        train_counts = np.zeros_like(y, dtype=np.int64)

        # define cross validation scheme
        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        # make indexes for train/validation subjects for each fold
        for i, (train_index, val_index) in enumerate(kf.split(X, y)):
            # print(f"bootstrap={b}/{n_bootstraps} {metric} Split {i + 1} - Training on {len(train_index)} samples, Testing on {len(val_index)} samples")

            X_train = X.iloc[train_index].copy()
            y_train = y[train_index].copy()
            X_val = X.iloc[val_index].copy()
            y_val = y[val_index].copy()

            if bootstrap:
                # Bootstrap the training data only
                bootstrap_indices = rng.choice(len(train_index), size=len(train_index), replace=True)
                X_train_boot = X_train.iloc[bootstrap_indices].reset_index(drop=True)
                y_train_boot = y_train[bootstrap_indices]
            else:
                X_train_boot = X_train
                y_train_boot = y_train

            X_train_boot_harmonized, X_val_harmonized = covbat_harmonize(X_train_boot, X_val)

            if set_parameters_manually == 0:
                # Fit model to train set
                print("fitting")
                opt.fit(X_train_boot_harmonized, y_train_boot)

                # Use model to predict on validation set
                val_predictions[val_index] = opt.predict(X_val_harmonized)

                # Predict for train set
                train_predictions[train_index] += opt.predict(X_train_boot_harmonized)

                model = opt.best_estimator_

            else:
                # Fit xgboost model using specified parameters
                xgb.fit(X_train_boot_harmonized, y_train_boot)
                model = xgb
                val_predictions[val_index] = xgb.predict(X_val_harmonized)
                train_predictions[train_index] += xgb.predict(X_train_boot_harmonized)

            # Store importances
            feature_importance_list.append(model.feature_importances_)

            # Keep track of the number of times that each subject is included in the train set
            train_counts[train_index] += 1

        # Correct the predictions for the train set by the number of times they appeared in the train set
        train_predictions /= train_counts

        if set_parameters_manually == 0:
            best_params = opt.best_params_
            print(f"Best Parameters for Final {metric} Model: {best_params}")
        elif set_parameters_manually == 1:
            best_params = params

        # Compute R2
        r2_val = r2_score(y, val_predictions)
        r2_train = r2_score(y, train_predictions)

        print(f"R2val = {r2_val:.3f}")

        # Calculate and print time it took to complete all model creations and predictions across all cv splits
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60.0
        print(f"XGB Bootstrap {b + 1}/{n_bootstraps} complete. Time since beginning of program: {elapsed_time:.2f} minutes")

        if bootstrap == 0:
            write_modeling_data_and_outcome_to_file(run_dummy_quick_fit, metric, params, set_parameters_manually, target, X,
                                                 r2_train, r2_val, best_params, bootstrap, elapsed_time)

        # plot_xgb_actual_vs_pred(metric, target, r2_train, r2_val, df, best_params, show_results_plot)
        r2_val_all_bootstraps.append(r2_val)

    r2_val_array_xgb = np.array(r2_val_all_bootstraps)

    feature_names = ['Sex'] + X.drop(columns=['Site', 'Sex']).columns.tolist()
    feature_importance_df = aggregate_feature_importances(feature_importance_list, feature_names, n_bootstraps,
                outputfilename=f"{target}_{metric}_{n_bootstraps}_xgb_feature_importance.txt", top_n=25)

    if set_parameters_manually == 1 & include_asd_in_train==0:
        X_train_fullsample = X
        y_train_fullsample = y
        X_train_fullsample_harmonized, X_test_harmonized = covbat_harmonize(X_train_fullsample, X_test)

        xgb.fit(X_train_fullsample_harmonized, y_train_fullsample)
        model_fullsample = xgb
        test_predictions = xgb.predict(X_test_harmonized)
        train_fullsample_predictions= xgb.predict(X_train_fullsample_harmonized)
        r2_test = r2_score(y_test, test_predictions)
        r2_train_fullsample = r2_score(y_train_fullsample, train_fullsample_predictions)
        print(f"R2train_fullset = {r2_train_fullsample:.3f}, R2test (ASD+group) = {r2_test}")


    return r2_val_array_xgb, feature_importance_df