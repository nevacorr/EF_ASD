import os
import matplotlib.pyplot as plt
from Utility_Functions_XGBoost import plot_correlations, remove_collinearity, plot_xgb_actual_vs_pred
from Utility_Functions_XGBoost import write_modeling_data_and_outcome_to_file, calculate_percentile
from Utility_Functions_XGBoost import plot_r2_distribution
from predict_SA_ridge import tune_ridge_alpha, predict_SA_ridge
from skopt import BayesSearchCV
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import numpy as np
import warnings
from load_data_for_ML import load_data
from predict_SA_xgboost import predict_SA_xgboost

# Suppress all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

target = "BRIEF2_GEC_T_score"
metric = 'subcort'
include_group = 1
n_bootstraps = 100
run_dummy_quick_fit = 0
alpha=0.05

# Define parameter ranges to be used (ranges if BayesCV will be used)
params = {"n_estimators": 353,  # (50, 2001),# Number of trees to create during training
          "min_child_weight": 11,
          # (1,11) # the number of samples required in each child node before attempting to split further
          "gamma": 5,
          # (0.01, 5.0, "log-uniform"),# regularization. Low values allow splits as long as they improve the loss function, no matter how small
          "eta": 0.011875,  # (0.005, 0.5, "log-uniform"),# learning rate
          "subsample": 1.0,  # (0.2, 1.0),# Fraction of training dta that is sampled for each boosting round
          "colsample_bytree": 1.0,  # (0.2, 1.0)  the fraction of features to be selected for each tree
          "max_depth": 3  # (2, 6), }#maximum depth of each decision tree
          }

X, y, df = load_data(target, metric, include_group, run_dummy_quick_fit, show_heat_map=0, remove_colinear=0)

r2_test_xgboost = predict_SA_xgboost(X, y, df, target, metric, params, run_dummy_quick_fit, set_params_man=1,
                show_results_plot=0, n_bootstraps=n_bootstraps)

r2_test_array_xgb = np.array(r2_test_xgboost)

result_text, percentile_value = calculate_percentile(r2_test_array_xgb, alpha)

plot_r2_distribution(r2_test_array_xgb, result_text, percentile_value, target, metric,
                     alpha, n_bootstraps, alg='XGBoost')

tune_ridge_alpha(X, y)

plt.show()