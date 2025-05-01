import os
import matplotlib.pyplot as plt
from Utility_Functions_XGBoost import plot_correlations, remove_collinearity, plot_xgb_actual_vs_pred
from Utility_Functions_XGBoost import write_modeling_data_and_outcome_to_file
from skopt import BayesSearchCV
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import numpy as np
import time
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score
from load_brain_data import load_subcortical_data, load_and_clean_volume_data, load_and_clean_dti_data
import itertools
from neurocombat_sklearn import CombatModel

def load_data(target, metric, include_group_feature, run_dummy_quick_fit, show_heat_map, remove_colinear):

    print(f"Running with target = {target} metric = {metric} include_group = {include_group_feature} "
          f"quick fit = {run_dummy_quick_fit}")

    target = target
    metric = metric
    show_correlation_heatmap = show_heat_map
    remove_collinear_features = remove_colinear
    include_group_feature = include_group_feature

    # Define directories to be used
    working_dir = os.getcwd()
    vol_dir = "/Users/nevao/R_Projects/IBIS_EF/"
    volume_datafilename = "final_df_for_xgboost.csv"

    if metric in {"fa_VSA", "md_VSA", "ad_VSA", "rd_VSA" }:
        dti_dir = ("/Users/nevao/Documents/Genz/source_data/updated imaging_2-27-25/"
                   "IBISandDS_VSA_DTI_Siemens_CMRR_v02.02_20250227/Siemens_CMRR/")
        ad_datafilename = "IBISandDS_VSA_DTI_SiemensAndCMRR_FiberAverage_AD_v02.02_20250227.csv"
        fa_datafilename = "IBISandDS_VSA_DTI_SiemensAndCMRR_FiberAverage_FA_v02.02_20250227.csv"
        md_datafilename = "IBISandDS_VSA_DTI_SiemensAndCMRR_FiberAverage_MD_v02.02_20250227.csv"
        rd_datafilename = "IBISandDS_VSA_DTI_SiemensAndCMRR_FiberAverage_RD_v02.02_20250227.csv"
        data_filenames = {
            "fa_VSA": fa_datafilename,"ad_VSA": ad_datafilename,"md_VSA": md_datafilename,"rd_VSA": rd_datafilename
        }
        datafilename = data_filenames.get(metric)
        df = load_and_clean_dti_data(dti_dir, datafilename, vol_dir, volume_datafilename, target, include_group_feature)
    elif metric == "volume":
        datafilename = volume_datafilename
        df = load_and_clean_volume_data(vol_dir, datafilename, target, include_group_feature)
    elif metric == "subcort":
        datafilename = volume_datafilename
        subcort_dir = '/Users/nevao/Documents/Genz/source_data/IBIS1&2_volumes_v3.13'
        df = load_subcortical_data(subcort_dir, vol_dir, datafilename, target,  include_group_feature)
        df = df.reset_index(drop=True)
    if run_dummy_quick_fit == 1:
        df = df.sample(frac=0.1, random_state=42)
        n_iter = 5

    if show_correlation_heatmap:
        # plot feature correlation heatmap
        plot_title = f"Correlation between regional {metric}"
        corr_matrix = plot_correlations(df, target, plot_title)
        plt.show()

    if remove_collinear_features:
        # remove features so that none have more than 0.9 correlation with other
        df = remove_collinearity(df, 0.9)
        plot_title="After removing colinear features"
        corr_matrix = plot_correlations(df, target, plot_title)
        plt.show()

    # Make matrix of features
    X = df.drop(columns=[target])

    # Make vector with target value
    y = df[target].values

    return X, y, df

# Main xgboost prediction code
def predict_SA_xgboost(df, X, y, target, metric, params, run_dummy_quick_fit, set_params_man, show_results_plot, bootstrap):

    set_parameters_manually = set_params_man
    show_results_plot = show_results_plot

    # set number of iterations for BayesCV
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

    # define cross validation scheme
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # make variables to hold predictions for train and test run, as well as counts for how many times each subject appears in a train set
    train_predictions = np.zeros_like(y, dtype=np.float64)
    test_predictions = np.zeros_like(y, dtype=np.float64)
    train_counts = np.zeros_like(y, dtype=np.int64)

    # record start time
    start_time = time.time()
    # make indexes for train/test subjects for each fold
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        print(f"{metric} Split {i + 1} - Training on {len(train_index)} samples, Testing on {len(test_index)} samples")

        # initialize combat model
        combat = CombatModel()

        X_train = X.iloc[train_index].copy()
        X_test = X.iloc[test_index].copy()

        # Create Categorical object for the training set sites
        train_sites = pd.Categorical(X_train['Site'])

        # Convert training sites to numeric codes (for harmonization)
        sites_train = train_sites.codes

        # Replace the 'Site' column in X_train with the codes
        X_train['Site'] = sites_train

        # Apply the same categorical mapping to the test set sites
        test_sites = pd.Categorical(X_test['Site'], categories=train_sites.categories)
        sites_test = test_sites.codes

        # Replace the 'Site' column in X_test with the codes
        X_test['Site'] = sites_test

        #  Replace NaN values with column mean for harmonization
        # Drop the 'Site' column because X_train and X_test after running combat will not have this column
        nan_indices_train = X_train.isna().drop(columns=['Site','Sex'])
        nan_indices_test = X_test.isna().drop(columns=['Site', 'Sex'])
        X_train_temp = X_train.copy()  # Create a copy of the training data to avoid modifying original data
        X_test_temp = X_test.copy()
        X_train_temp = X_train_temp.fillna(X_train_temp.mean()) # Replace NaN with the column mean (change to use scikit learn)
        X_test_temp = X_test_temp.fillna(X_train_temp.mean()) # Replace NaN with the column mean of the train set

        # Keep a copy of Sex
        sex_train = X_train_temp['Sex'].values.reshape(-1,1)
        sex_test = X_test_temp['Sex'].values.reshape(-1, 1)

        # Harmonize the training data
        X_train_combat = combat.fit_transform(X_train_temp.drop(columns=['Site', 'Sex']), sites_train.reshape(-1, 1))
        # Replace the original Nan values
        X_train_combat[nan_indices_train] = np.nan

        # Harmonize the test data (using the same harmonization model fitted on the training data)
        X_test_combat = combat.transform(X_test_temp.drop(columns=['Site', 'Sex']), sites_test.reshape(-1, 1))
        # Replace the original Nan values
        X_test_combat[nan_indices_test] = np.nan

        # Add sex values back into array for xgboost now that the brain measures have been harmonized
        X_train_combat = np.hstack([sex_train,X_train_combat])
        X_test_combat = np.hstack([sex_test, X_test_combat])

        if set_parameters_manually == 0:
            # Fit model to train set
            print("fitting")
            opt.fit(X_train_combat, y[train_index])

            # Use model to predict on test set
            test_predictions[test_index] = opt.predict(X_test_combat)

            # Predict for train set
            train_predictions[train_index] += opt.predict(X_train_combat)

        else:
            # Fit xgboost model using specified parameters
            xgb.fit(X_train_combat, y[train_index])
            test_predictions[test_index] = xgb.predict(X_test_combat)
            train_predictions[train_index] += xgb.predict(X_train_combat)

        # Keep track of the number of times that each subject is included in the train set
        train_counts[train_index] += 1

        # Calculate and print time it took to run this fold
        fold_time = time.time()
        elapsed_time = (fold_time - start_time) / 60.0
        print(f"{metric}  Time elapsed: {elapsed_time:.2f} minutes")

    # Correct the predictions for the train set by the number of times they appeared in the train set
    train_predictions /= train_counts

    # Put train and test_predictions in a dataframe
    df["test_predictions"]  = test_predictions
    df["train_predictions"] = train_predictions

    # Calculate and print time it took to complete all model creations and predictions across all cv splits
    end_time = time.time()
    elapsed_time = (end_time - start_time)  / 60.0
    print(f"{metric} Computations complete. Time to run all 10 folds: {elapsed_time:.2f} minutes")

    if set_parameters_manually == 0:
        best_params = opt.best_params_
        print(f"Best Parameters for Final {metric} Model: {best_params}")
    elif set_parameters_manually == 1:
        best_params = params

    # Compute R2
    r2_test = r2_score(df[target], df["test_predictions"])
    r2_train = r2_score(df[target], df["train_predictions"])

    print(f"Final performance. R2train = {r2_train:.3f} R2test = {r2_test:.3f}")

    write_modeling_data_and_outcome_to_file(run_dummy_quick_fit, metric, params, set_parameters_manually, target, df,
                                            r2_train, r2_test, best_params, bootstrap, elapsed_time)

    plot_xgb_actual_vs_pred(metric, target, r2_train, r2_test, df, best_params, show_results_plot)

 ######## End of function predict_SA_xgboost#######
 #####################################################################################################################

targets = ["BRIEF2_GEC_T_score"]
metrics = ['subcort']
include_group_options = 1

# Define parameter ranges to be used (ranges if BayesCV will be used)
params = {"n_estimators": 50,  # (50, 2001),# Number of trees to create during training
          "min_child_weight": 11,
          # (1,11) # the number of samples required in each child node before attempting to split further
          "gamma": 0.012,
          # (0.01, 5.0, "log-uniform"),# regularization. Low values allow splits as long as they improve the loss function, no matter how small
          "eta": 0.06649,  # (0.005, 0.5, "log-uniform"),# learning rate
          "subsample": 1.0,  # (0.2, 1.0),# Fraction of training dta that is sampled for each boosting round
          "colsample_bytree": 0.86,  # (0.2, 1.0)  the fraction of features to be selected for each tree
          "max_depth": 5  # (2, 6), }#maximum depth of each decision tree
          }

for target, metric, include_group in itertools.product(targets, metrics, include_group_options):

    run_dummy_quick_fit = 0

    X, y, df = load_data(
                target,
                metric,
                include_group,
                run_dummy_quick_fit,
                show_heat_map=0,
                remove_colinear=0,
)
    predict_SA_xgboost(
                X,
                y,
                df,
                target,
                metric,
                params,
                run_dummy_quick_fit,
                set_params_man=1,
                show_results_plot=0,
                bootstrap=0)