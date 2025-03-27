import pandas as pd
import os
import matplotlib.pyplot as plt
from Utility_Functions_XGBoost import load_and_clean_data, plot_correlations, remove_collinearity, plot_xgb_actual_vs_pred
from Utility_Functions_XGBoost import load_and_clean_dti_data, write_modeling_data_and_outcome_to_file
from skopt import BayesSearchCV
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import numpy as np
import time
import pickle
from sklearn.metrics import mean_squared_error,r2_score

target = "Flanker_Standard_Age_Corrected"
metric = "md"
run_training = 1
set_parameters_manually = 0
show_correlation_heatmap = 0
remove_collinear_features = 0
include_group_feature = 0

working_dir = os.getcwd()

if metric == "volume":
    vol_dir = "/Users/neva o/R_Projects/IBIS_EF/"
    volume_datafilename = "final_df_for_xgboost.csv"  
elif metric in {"fa", "md", "ad", "rd"}:
    dti_dir = ("/Users/nevao/Documents/Genz/source_data/updated imaging_2-27-25/"
               "IBISandDS_VSA_DTI_Siemens_CMRR_v02.02_20250227/Siemens_CMRR/")
    ad_datafilename = "IBISandDS_VSA_DTI_SiemensAndCMRR_FiberAverage_AD_v02.02_20250227.csv"
    fa_datafilename = "IBISandDS_VSA_DTI_SiemensAndCMRR_FiberAverage_FA_v02.02_20250227.csv"
    md_datafilename = "IBISandDS_VSA_DTI_SiemensAndCMRR_FiberAverage_MD_v02.02_20250227.csv"
    rd_datafilename = "IBISandDS_VSA_DTI_SiemensAndCMRR_FiberAverage_RD_v02.02_20250227.csv"

if run_training:
     # load and clean data
    if metric == "volume":
         df = load_and_clean_data(vol_dir, volume_datafilename, target, include_group_feature)

    if metric in {"fa", "md", "ad", "rd" }:
        data_filenames = {
            "fa": fa_datafilename,"ad": ad_datafilename,"md": md_datafilename,"rd": rd_datafilename
        }
        datafilename = data_filenames.get(metric)
        df = load_and_clean_dti_data(dti_dir, datafilename, vol_dir, volume_datafilename, target, include_group_feature)

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

    X = df.drop(columns=[target]).values
    y = df[target].values

    # params = {"n_estimators": (100, 500),# Number of trees to create during training
    #           "min_child_weight": (1,5), # the number of samples required in each child node before attempting to split further
    #           "gamma": (0.01, 2.0, "log-uniform"),# regularization. Low values allow splits as long as they improve the loss function, no matter how small
    #           "eta": (0.05, 0.2, "log-uniform"),# learning rate
    #           "subsample": (0.2, 0.8),# Fraction of training dta that is sampled for each boosting round
    #           "colsample_bytree": (0.2, 1.0),#the fraction of features to be selected for each tree
    #           "max_depth": (3, 5), }#maximum depth of each decision tree

    params = {"n_estimators": (50, 2001),
              "min_child_weight": (1, 11),
              "gamma": (0.01, 5.0, "log-uniform"),
              "eta": (0.005, 0.5, "log-uniform"),
              "subsample": (0.2, 1.0),
              "colsample_bytree": (0.2, 1.0),
              "max_depth": (2, 6), }

    if set_parameters_manually == 0:

        xgb = XGBRegressor(objective="reg:squarederror", n_jobs=8)
        opt = BayesSearchCV(xgb, params, n_iter=100, n_jobs=8)

    else:
        xgb = XGBRegressor(
            objective="reg:squarederror",
            n_jobs=16,
            colsample_bytree=1.0,#the fraction of features to be selected for each tree
            eta=0.00847,  # learning rate
            gamma=0.4,  # regularization. Low values allow splits as long as they improve the loss function, no matter how small
            max_depth=2,#maximum depth of each decision tree
            min_child_weight=1,# the number of samples required in each child node before attempting to split further
            n_estimators=50,  # Number of trees to create during training
            subsample=1.0  # Fraction of training dta that is sampled for each boosting round
        )

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    train_predictions = np.zeros_like(y, dtype=np.float64)
    test_predictions = np.zeros_like(y, dtype=np.float64)
    train_counts = np.zeros_like(y, dtype=np.int64)

    start_time = time.time()
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        print(f"{metric} Split {i + 1} - Training on {len(train_index)} samples, Testing on {len(test_index)} samples")

        if set_parameters_manually == 0:
            # Fit model to train set
            print("fitting")
            opt.fit(X[train_index], y[train_index])

            # Use model to predict on test set
            print(f"predicting {metric} for test set")
            test_predictions[test_index] = opt.predict(X[test_index])

            # Predict for train set
            print(f"predicting {metric} for train set")
            train_predictions[train_index] += opt.predict(X[train_index])

        else:
            # Fit xgboost model using specified parameters
            xgb.fit(X[train_index], y[train_index])
            test_predictions[test_index] = xgb.predict(X[test_index])
            train_predictions[train_index] += xgb.predict(X[train_index])

        # Keep track of the number of times that each subject is included in the train set
        train_counts[train_index] += 1

        # Compute performance metrics for train set
        mse_train = mean_squared_error(y[train_index], test_predictions[train_index], squared=False)
        r2_train = r2_score(y[train_index], train_predictions[train_index])

        # Compute performance metrics for test set
        mse_test = mean_squared_error(y[test_index], test_predictions[test_index], squared=False)
        r2_test = r2_score(y[test_index], test_predictions[test_index])

        print(f"Performance train set performance for Split {i + 1}: R² = {r2_train:.4f}, MSE = {mse_train:.4f}")
        print(f"Performance test set performance for Split {i + 1}: R² = {r2_test:.4f}, MSE = {mse_test:.4f}")

        # Save model to file
        if set_parameters_manually == 0:
            # Save model to file
            if i==1:
                with open(f"{target}_{metric}_trained_model.pkl", "wb") as f:
                    pickle.dump(opt, f)
                print(f"First trained model saved to {target}_{metric}_trained_model.pkl")
        else:
            with open(f"{target}_{metric}_trained_model.pkl", "wb") as f:
                pickle.dump(xgb, f)
            print(f"Trained model saved to {target}_{metric}_trained_model.pkl")

    # Correct the predictions for teh train set by the number of times they appeared in the train set
    train_predictions /= train_counts

    # Put test_predictions in a dataframe column
    df["test_predictions"]  = test_predictions
    df["train_predictions"] = train_predictions

    # Calculate time it took to complete all model creations and predictions across all cv splits
    end_time = time.time()
    elapsed_time = (end_time - start_time)  / 60.0
    print(f"{metric} Computations complete. Time to run all 10 folds: {elapsed_time:.2f} minutes")

    # Save data and predictions to file
    df.to_csv(f"{target}_{metric}_cv_predictions.csv", index=False)

    if set_parameters_manually == 0:
        print(f"Best Parameters for Final {metric} Model: {opt.best_params_}")
        # Save best parameters for the final model to a text file
        with open(f"{target}_{metric}_best_params_final.txt", "w") as file:
            file.write(str(opt.best_params_))
        print(f"{metric} Best parameters saved to {target}_best_params_final.txt")

else:

    # Read data and predictions from file
    df = pd.read_csv(f"{target}_{metric}_cv_predictions.csv")

# Read model from file
model_filename = f"{target}_{metric}_trained_model.pkl"

with open(model_filename, "rb") as f:
    loaded_model = pickle.load(f)

# Compute R2
r2_test = r2_score(df[target], df["test_predictions"])
r2_train = r2_score(df[target], df["train_predictions"])

write_modeling_data_and_outcome_to_file(metric, params, set_parameters_manually, loaded_model, target, X, r2_train, r2_test)

plot_xgb_actual_vs_pred(metric, target, r2_train, r2_test, loaded_model, df)

mystop = 1