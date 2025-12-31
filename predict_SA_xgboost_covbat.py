from skopt import BayesSearchCV
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import numpy as np
import time
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from sklearn.model_selection import train_test_split
from Utility_Functions_XGBoost import write_modeling_data_and_outcome_to_file, aggregate_feature_importances
from covbat_harmonize import covbat_harmonize
from skopt.callbacks import VerboseCallback

def predict_SA_xgboost_covbat(X, y, group_vals, sex_vals, target, metric, params, run_dummy_quick_fit, set_params_man,
                       show_results_plot, bootstrap, n_bootstraps, X_test, y_test, include_asd_in_train):
    # X is features for all neurotypical subjects (to be used for training and validation)
    # X_test is features for all ASD subjects
    # y is target (EF) for neurotypical subjects
    # y_test is target (EF) for ASD subjects

    # Suppress all FutureWarnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    set_parameters_manually = set_params_man

    r2_val_all_bootstraps=[]

    # set number of iterations for BayesCV
    if run_dummy_quick_fit:
        n_iter = 5
    else:
        n_iter = 100

    # record start time
    start_time = time.time()
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility

    X_train_orig, X_val, y_train_orig, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Harmonize features for all neurotypical subjects, taking into account covariance patterns by site
    X_train_h, X_val_h, X_test_h = covbat_harmonize(X_train_orig, X_val, X_test)  # harmonize covariates

    # If using BayesSearchCV to find optimal xgb parameters
    if set_parameters_manually == 0: #if search for best parameters

        # Create xgb object without specifying parameters
        xgb = XGBRegressor(objective="reg:squarederror", n_jobs=-1)
        # Create BayesSearchCV  object using xfb object, specify 5 folds
        # This means that it will do 5 splits of the data for each fold for training and validation
        opt = BayesSearchCV(xgb, params, n_iter=n_iter, cv=5, n_jobs=-1, verbose=2)

        # Use VerboseCallback to print iteration number only
        callback = [VerboseCallback(n_total=n_iter)]

        # Perform BayesCV optimization
        opt.fit(X_train_h, y_train_orig, callback=callback)
        # Save the best parameters and print them to screen
        best_params = opt.best_params_
        print("Best hyperparameters for neurotypical group:", best_params)

        # Recreate xgb_final object, this time with the best parameters found with BayesSearchCV based on all neurotypical
        # subjects
        xgb_final = XGBRegressor(
            objective="reg:squarederror",
            n_jobs=16,
            **best_params)

    else:  # if parameters are to be set manually at fixed values

        # set xgb parameters, according to values sent to this function
        xgb_final = XGBRegressor(
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

        best_params = params

    # If not bootstrapping, only run xgboost once on neurotypical training set and once on validation set. This will give only
    # one R2 values; it will not give confidence intervals for R2
    if bootstrap == 0:
        n_bootstraps = 1 # force one iteration with no bootstrapping

    # Make variable to store xgb feature importance
    feature_importance_list = []

    for b in range(n_bootstraps):

        # Make variables to hold predictions for train and validation run, as well as counts for how many times each subject appears in a train set
        train_predictions = np.zeros_like(y, dtype=np.float64)
        val_predictions = np.zeros_like(y, dtype=np.float64)
        train_counts = np.zeros_like(y, dtype=np.int64)

        # define cross validation scheme
        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        # make indexes for train/validation subjects for each fold. Number of folds is specified above.
        for i, (train_index, val_index) in enumerate(kf.split(X_train_h, y_train_orig)):
            # print(f"bootstrap={b}/{n_bootstraps} {metric} Split {i + 1} - Training on {len(train_index)} samples, Validation set has {len(val_index)} samples")

            # Split the data into train and validation sets. This is not bootstrapping.
            # This just splits the data, one out of n_splits folds.
            X_train = X_train_h.iloc[train_index].copy()
            y_train = y_train_orig[train_index].copy()
            X_val = X_train_h.iloc[val_index].copy()
            y_val = y_train_orig[val_index].copy()

            if bootstrap:
                # Bootstrap (resample with replacement) the training data only.
                bootstrap_indices = rng.choice(len(train_index), size=len(train_index), replace=True)
                X_train_boot = X_train.iloc[bootstrap_indices].reset_index(drop=True)
                y_train_boot = y_train[bootstrap_indices]
            else:
                # Do not bootstrap the training data. Just use whole training set as it is, but assign
                # the name "X_train_boot" and "y_train_boot" to the training set features and target variables.
                X_train_boot = X_train
                y_train_boot = y_train

            # Fit xgboost model using xgb object created above
            xgb_final.fit(X_train_boot, y_train_boot)
            # Assign the name "model" to the xgb object
            model = xgb_final
            # Predict the target variable for the validation set for this fold
            val_predictions[val_index] = xgb_final.predict(X_val)
            # Predict teh target variable for train_boot for this fold
            train_predictions[train_index] += xgb_final.predict(X_train_boot)

            # Store feature importance for this fold and bootstrap sample
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
        r2_val = r2_score(y_val, val_predictions)
        r2_train = r2_score(y_train_boot, train_predictions)

        print(f"R2val = {r2_val:.3f}")

        # Calculate and print time it took to complete all model creations and predictions across all cv splits for this bootstrap
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

    if set_parameters_manually == 1 and not include_asd_in_train:
        X_train_fullsample = X
        y_train_fullsample = y
        X_train_fullsample_harmonized, X_test_harmonized = covbat_harmonize(X_train_fullsample, X_test)

        xgb_final.fit(X_train_fullsample_harmonized, y_train_fullsample)
        model_fullsample = xgb_final
        test_predictions = xgb_final.predict(X_test_harmonized)
        train_fullsample_predictions= xgb_final.predict(X_train_fullsample_harmonized)
        r2_test = r2_score(y_test, test_predictions)
        r2_train_fullsample = r2_score(y_train_fullsample, train_fullsample_predictions)
        print(f"R2train_fullset = {r2_train_fullsample:.3f}\n R2test (ASD+group) = {r2_test}")

    return r2_val_array_xgb, feature_importance_df
