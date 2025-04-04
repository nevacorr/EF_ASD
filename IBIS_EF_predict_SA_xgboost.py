import os
import matplotlib.pyplot as plt
from Utility_Functions_XGBoost import plot_correlations, remove_collinearity, plot_xgb_actual_vs_pred
from Utility_Functions_XGBoost import write_modeling_data_and_outcome_to_file
from skopt import BayesSearchCV
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import numpy as np
import time
from sklearn.metrics import mean_squared_error,r2_score
from load_brain_data import load_subcortical_data, load_and_clean_volume_data, load_and_clean_dti_data
import itertools

# Main xgboost prediction code
def predict_SA_xgboost(target, metric, params, include_group_feature, run_quick_fit, set_params_man, show_heat_map,
                       remove_colinear, show_results_plot):

    print(f"Running with target = {target} metric = {metric} include_group = {include_group_feature} "
          f"quick fit = {run_quick_fit}")

    target = target
    metric = metric
    run_dummy_quick_fit = run_quick_fit
    set_parameters_manually = set_params_man
    show_correlation_heatmap = show_heat_map
    remove_collinear_features = remove_colinear
    include_group_feature = include_group_feature
    show_results_plot = show_results_plot

    # set number of iterations for BayesCV
    n_iter = 100

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
    X = df.drop(columns=[target]).values

    # Make vector with target value
    y = df[target].values

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

        if set_parameters_manually == 0:
            # Fit model to train set
            print("fitting")
            opt.fit(X[train_index], y[train_index])

            # Use model to predict on test set
            test_predictions[test_index] = opt.predict(X[test_index])

            # Predict for train set
            train_predictions[train_index] += opt.predict(X[train_index])

        else:
            # Fit xgboost model using specified parameters
            xgb.fit(X[train_index], y[train_index])
            test_predictions[test_index] = xgb.predict(X[test_index])
            train_predictions[train_index] += xgb.predict(X[train_index])

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
                                            r2_train, r2_test, best_params, elapsed_time)

    plot_xgb_actual_vs_pred(metric, target, r2_train, r2_test, df, best_params, show_results_plot)

 ######## End of function predict_SA_xgboost#######
 #####################################################################################################################

targets = ["Flanker_Standard_Age_Corrected", "BRIEF2_GEC_raw_score","BRIEF2_GEC_T_score", "DCCS_Standard_Age_Corrected"]
metrics = ['subcort', 'volume', 'fa_VSA', 'md_VSA', 'rd_VSA']
include_group_options = [0, 1]

# Define parameter ranges to be used (ranges if BayesCV will be used)
params = {"n_estimators": (50, 2001),  # (50, 2001),# Number of trees to create during training
          "min_child_weight": (1, 11),
          # (1,11) # the number of samples required in each child node before attempting to split further
          "gamma": (0.01, 5.0, "log-uniform"),
          # (0.01, 5.0, "log-uniform"),# regularization. Low values allow splits as long as they improve the loss function, no matter how small
          "eta": (0.005, 0.5, "log-uniform"),  # (0.005, 0.5, "log-uniform"),# learning rate
          "subsample": (0.2, 1.0),  # (0.2, 1.0),# Fraction of training dta that is sampled for each boosting round
          "colsample_bytree": (0.2, 1.0),  # (0.2, 1.0)  the fraction of features to be selected for each tree
          "max_depth": (2, 6)  # (2, 6), }#maximum depth of each decision tree
          }

for target, metric, include_group in itertools.product(targets, metrics, include_group_options):
    predict_SA_xgboost(
        target,
        metric,
        params,
        include_group,
        run_quick_fit=0,
        set_params_man=0,
        show_heat_map=0,
        remove_colinear=0,
        show_results_plot=0)