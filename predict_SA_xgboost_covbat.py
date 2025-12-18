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
from Utility_Functions_XGBoost import plot_shap_magnitude_kde, impute_by_site_median_with_nan_indices
import os
os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources"
import rpy2.robjects as robjects
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro

# Source your R wrapper script
robjects.r('source("~/R_Projects/IBIS_EF_xgboost/covbat_wrapper_for_use_with_python.R")')

# np.int = int   #patch because of bug in numpy version that affects gridsearchCV

def predict_SA_xgboost_covbat(X, y, group_vals, sex_vals, target, metric, params, run_dummy_quick_fit, set_params_man,
                       show_results_plot, bootstrap, n_bootstraps):

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

            # Assign covbat functions
            fit_covbat = robjects.r['fit_covbat']
            apply_covbat = robjects.r['apply_covbat']

            #  Replace NaN values with column median for harmonization
            fcols = X_train_boot.columns.difference(['Site', 'Sex'])

            X_train_boot_temp, X_val_temp, nan_indices_train, nan_indices_val = impute_by_site_median_with_nan_indices(
                X_train_boot,
                X_val,
                feature_cols=fcols,
                site_col='Site'
            )

            # Keep a copy of Sex
            sex_train = X_train_boot_temp['Sex'].values.reshape(-1,1)
            sex_val = X_val_temp['Sex'].values.reshape(-1, 1)

            # --- Convert to R data frames ---
            with localconverter(robjects.default_converter + pandas2ri.converter):
                X_train_r = robjects.conversion.py2rpy(X_train_boot_temp.drop(columns=['Site', 'Sex']))
                X_val_r = robjects.conversion.py2rpy(X_val_temp.drop(columns=['Site', 'Sex']))

            batch_train_r = robjects.FactorVector(X_train_boot_temp['Site'])
            batch_val_r = robjects.FactorVector(X_val_temp['Site'])

            # --- Fit CovBat on training data ---
            covbat_fit = fit_covbat(X_train_r, batch_train_r)

            # --- Apply CovBat ---
            X_train_boot_harmonized_r = apply_covbat(covbat_fit, X_train_r, batch_train_r)
            X_val_harmonized_r = apply_covbat(covbat_fit, X_val_r, batch_val_r)

            # --- Convert back to pandas ---
            with localconverter(robjects.default_converter + pandas2ri.converter):
                X_train_boot_harmonized = robjects.conversion.rpy2py(X_train_boot_harmonized_r)
                X_val_harmonized = robjects.conversion.rpy2py(X_val_harmonized_r)

            # --- Restore NaNs ---
            X_train_boot_harmonized[nan_indices_train] = np.nan
            X_val_harmonized[nan_indices_val] = np.nan

            # Add sex values back into array for xgboost now that the brain measures have been harmonized
            X_train_boot_harmonized = np.hstack([sex_train,X_train_boot_harmonized])
            X_val_harmonized = np.hstack([sex_val, X_val_harmonized])

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

                # explain the GAM model with SHAP
                explainer= shap.Explainer(xgb, X_train_boot_harmonized)
                shap_values = explainer(X_val_harmonized)
                shap_feature_names = list(X_train_boot.drop(columns="Site"))

                # Save SHAP values and metadata
                if b == 0 and i == 0 and n_bootstraps==1 and set_parameters_manually == 1:
                    all_shap_values = shap_values.values
                    all_feature_values = X.iloc[val_index].copy()
                    all_group_labels = group_vals.iloc[val_index].copy()
                    all_sex_labels = sex_vals.iloc[val_index].copy()
                elif set_parameters_manually == 1 and n_bootstraps ==1:
                    all_shap_values = np.vstack([all_shap_values, shap_values.values])
                    all_feature_values = pd.concat(
                        [all_feature_values, X.iloc[val_index].copy().reset_index(drop=True)],
                        axis=0
                    ).reset_index(drop=True)
                    all_group_labels = pd.concat(
                        [all_group_labels, group_vals.iloc[val_index].copy().reset_index(drop=True)],
                        axis=0
                    ).reset_index(drop=True)
                    all_sex_labels = pd.concat(
                            [all_sex_labels, sex_vals.iloc[val_index].copy().reset_index(drop=True)],
                        axis=0
                    ).reset_index(drop=True)

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

    if set_parameters_manually == 1 and n_bootstraps == 1:
        # all_shap_values shape: (n_samples, n_features)
        mean_abs_shap = np.abs(all_shap_values).mean(axis=0)

        # Create a DataFrame for easier handling and plotting
        shap_feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)

        # plot_top_shap_distributions_by_group(
        #     shap_feature_importance_df,
        #     all_shap_values,
        #     all_group_labels,
        #     all_sex_labels,
        #     feature_names,
        #     top_n=20
        # )

        plot_shap_magnitude_histograms_equal_bins(
            all_shap_values,
            all_group_labels,
            all_sex_labels,
            feature_names,
            sex_feature_name='Sex'
        )

        plot_shap_magnitude_by_sex_and_group(
            all_shap_values,
            all_group_labels,
            all_sex_labels,
            feature_names,
            sex_feature_name='Sex'
        )

        plot_shap_magnitude_kde(
            all_shap_values,
            all_group_labels,
            all_sex_labels,
            feature_names,
            sex_feature_name='Sex'
        )

    return r2_val_array_xgb, feature_importance_df
