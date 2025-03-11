import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from Utility_Functions_XGBoost import load_and_clean_data, plot_correlations, remove_collinearity
from skopt import BayesSearchCV
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import numpy as np
import time
import pickle
from sklearn.metrics import mean_squared_error,r2_score

target = "BRIEF2_GEC_T_score"
run_training = 0
remove_collinear_features = 0

working_dir = os.getcwd()

datafilename = "final_df_for_xgboost.csv"

if run_training:

    # load and clean data
    df = load_and_clean_data(working_dir, datafilename, target)

    # plot feature correlation heatmap
    plot_title="Correlation between regional volume measures after dividing by totTissue"
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

        params = {"n_estimators": (50, 2001),
                  "min_child_weight": (1, 11),
                  "gamma": (0.01, 5.0, "log-uniform"),
                  "eta": (0.005, 0.5, "log-uniform"),
                  "subsample": (0.2, 1.0),
                  "colsample_bytree": (0.2, 1.0),
                  "max_depth": (2, 6), }

        xgb = XGBRegressor(objective="reg:squarederror", nthread=16)
        opt = BayesSearchCV(xgb, params, n_iter=100, n_jobs=-1)

        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        predictions = np.zeros_like(y, dtype=np.float64)

        start_time = time.time()
        for i, (train_index, test_index) in enumerate(kf.split(X, y)):
            print(f"Split {i + 1} - Training on {len(train_index)} samples, Testing on {len(test_index)} samples")

            # Fit model to train set
            print("fitting")
            opt.fit(X[train_index], y[train_index])

            # Use model to predict on test set
            print("predicting")
            predictions[test_index] = opt.predict(X[test_index])

            # Calculate and print elapsed time since program start
            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60.0
            print(f"Elapsed time for fold {i+1}: {elapsed_time:.2f} minutes")

            # Compute performance metrics
            mse = mean_squared_error(y[test_index], predictions[test_index], squared=False)
            r2 = r2_score(y[test_index], predictions[test_index])

            print(f"Best Parameters for Split {i + 1}: {opt.best_params_}")
            print(f"Performance for Split {i + 1}: R² = {r2:.4f}, MSE = {mse:.4f}")

            # Save model to file
            with open(f"{target}_{i+1}_trained_model.pkl", "wb") as f:
                pickle.dump(opt, f)
            print(f"Trained model saved to {target}_{i+1}_trained_model.pkl")

        # Put predictions in a dataframe column
        df["Predictions"]  = predictions

        # Calculate time it took to complete all predictions across all cv splits
        end_time = time.time()
        elapsed_time = (end_time - start_time)  / 60.0
        print(f"Computations complete. Time to run all 10 folds: {elapsed_time:.2f} minutes")

        # Save predictions to file
        df.to_csv(f"{target}_cv_predictions.csv", index=False)

else:

    # Read predictions from file
    df = pd.read_csv(f"{target}_cv_predictions.csv")

    # Read model from first split from file
    model_filename = f"{target}_1_trained_model.pkl"

    with open(model_filename, "rb") as f:
        loaded_model = pickle.load(f)

# Compute R²
r2_final = r2_score(df[target], df["Predictions"])

# Plot actual vs. predicted values
plt.figure(figsize=(10,8))
sns.scatterplot(x=df[target] , y=df["Predictions"])

# Plot line where actual-predicted
y=df[target]
# plot y = x line
plt.plot([min(y), max(y)], [min(y), max(y)], linestyle = "--", color="red")
plt.xlabel(f"Actual {target}")
plt.ylabel(f"Predicted {target}")
plt.title(f"Predicted vs. Actual {target}\n Coefficient of Determination: {r2_final:.2f}")
plt.show()



mystop = 1