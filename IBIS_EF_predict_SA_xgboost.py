import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from Utility_Functions_XGBoost import load_and_clean_data, plot_correlations, remove_collinearity
from Utility_Functions_XGBoost import load_and_clean_dti_data
from skopt import BayesSearchCV
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import numpy as np
import time
import pickle
from sklearn.metrics import mean_squared_error,r2_score

target = "Flanker_Standard_Age_Corrected"
metric = "fa"
run_training = 1
set_parameters_manually = 0
show_correlation_heatmap = 1
remove_collinear_features = 0
include_group_feature = 0

working_dir = os.getcwd()
vol_dir = "/Users/nevao/R_Projects/IBIS_EF/"
dti_dir = ("/Users/nevao/Documents/Genz/source_data/updated imaging_2-27-25/"
           "IBISandDS_VSA_DTI_Siemens_CMRR_v02.02_20250227/Siemens_CMRR/")

volume_datafilename = "final_df_for_xgboost.csv"
ad_datafilename = "IBISandDS_VSA_DTI_SiemensAndCMRR_FiberAverage_AD_v02.02_20250227.csv"
fa_datafilename = "IBISandDS_VSA_DTI_SiemensAndCMRR_FiberAverage_FA_v02.02_20250227.csv"
md_datafilename = "IBISandDS_VSA_DTI_SiemensAndCMRR_FiberAverage_MD_v02.02_20250227.csv"
rd_datafilename = "IBISandDS_VSA_DTI_SiemensAndCMRR_FiberAverage_RD_v02.02_20250227.csv"

for metric in ["fa", "md", "ad", "rd"]:

    if run_training:

        # load and clean data
        # df = load_and_clean_data(vol_dir, volume_datafilename, target, include_group_feature)

        if metric == "fa":
            datafilename = fa_datafilename
        elif metric == "ad":
            datafilename = ad_datafilename
        elif metric == "md":
            datafilename = md_datafilename
        elif metric == "rd":
            datafilename = rd_datafilename

        df = load_and_clean_dti_data(dti_dir, datafilename, vol_dir, volume_datafilename, target, include_group_feature)

        if show_correlation_heatmap:
            # plot feature correlation heatmap
            # plot_title="Correlation between regional volume measures after dividing by totTissue"
            plot_title = "Correlation between regional fa"
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
                eta=0.05,  # learning rate
                gamma=0.1,  # regularization. Low values allow splits as long as they improve the loss function, no matter how small
                max_depth=6,#maximum depth of each decision tree
                min_child_weight=1,# the number of samples required in each child node before attempting to split further
                n_estimators=200,  # Number of trees to create during training
                subsample=0.8  # Fraction of training dta that is sampled for each boosting round
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

                xgb.fit(X[train_index], y[train_index])
                test_predictions[test_index] = xgb.predict(X[test_index])
                train_predictions[train_index] += xgb.predict(X[train_index])

            train_counts[train_index] += 1

            # Calculate and print elapsed time since program start
            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60.0
            print(f"Elapsed time for fold {i+1}: {elapsed_time:.2f} minutes")

            # Compute performance metrics for train set
            mse_train = mean_squared_error(y[train_index], test_predictions[train_index], squared=False)
            r2_train = r2_score(y[train_index], train_predictions[train_index])

            # Compute performance metrics for test set
            mse_test = mean_squared_error(y[test_index], test_predictions[test_index], squared=False)
            r2_test = r2_score(y[test_index], test_predictions[test_index])

            if set_parameters_manually == 0:
                print(f"Best Parameters for Split {i + 1}: {opt.best_params_}")
            print(f"Performance train set performance for Split {i + 1}: R² = {r2_train:.4f}, MSE = {mse_train:.4f}")
            print(f"Performance test set performance for Split {i + 1}: R² = {r2_test:.4f}, MSE = {mse_test:.4f}")

            if set_parameters_manually == 0:
                # Save model to file
                with open(f"{target}_{i+1}_trained_model.pkl", "wb") as f:
                    pickle.dump(opt, f)
                print(f"Trained model saved to {target}_{i+1}_trained_model.pkl")
            else:
                with open(f"{target}_{i+1}_trained_model.pkl", "wb") as f:
                    pickle.dump(xgb, f)
                print(f"Trained model saved to {target}_{i+1}_trained_model.pkl")

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
        df = pd.read_csv(f"{target}_cv_predictions.csv")

    # Read model from first split from file
    model_filename = f"{target}_{metric}_1_trained_model.pkl"

    with open(model_filename, "rb") as f:
        loaded_model = pickle.load(f)

    # Compute R2
    r2_test = r2_score(df[target], df["test_predictions"])
    r2_train = r2_score(df[target], df["train_predictions"])

    # Create subplots with 2 rows and 1 column
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    # Define a list of prediction types and corresponding R2 values
    prediction_data = [
        ("test", r2_test, df["test_predictions"]),
        ("train", r2_train, df["train_predictions"])
    ]

    colsample_bytree=loaded_model.colsample_bytree
    n_estimators = loaded_model.n_estimators
    min_child_weight = loaded_model.min_child_weight
    gamma = loaded_model.gamma
    eta = loaded_model.kwargs['eta']
    subsample = loaded_model.subsample
    max_depth = loaded_model.max_depth

    for i, (data_type, r2, predictions) in enumerate(prediction_data):
        sns.scatterplot(x=df[target], y=predictions, ax=axes[i])
        axes[i].plot([min(df[target]), max(df[target])], [min(df[target]), max(df[target])], linestyle="--", color="red")
        axes[i].set_xlabel(f"Actual {metric} {target}")
        axes[i].set_ylabel(f"Predicted {metric} {target}")
        axes[i].set_title(f"{data_type.capitalize()} {metric} Predictions\nR2: {r2:.2f}\n colsample_by_tree={colsample_bytree} "
                          f"n_estimators={n_estimators} min_child_weight={min_child_weight} gamma={gamma}\n eta={eta} "
                          f"subsample={subsample} max_depth={max_depth}")

    # Show the plot
    plt.tight_layout()
    plt.savefig(f"metric_xgboost_actual vs predicted")
    plt.show(block=False)


mystop = 1