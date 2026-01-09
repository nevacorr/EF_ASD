import matplotlib.pyplot as plt
from Utility_Functions_XGBoost import calculate_percentile
from Utility_Functions_XGBoost import plot_r2_distribution
from predict_SA_ridge import tune_ridge_alpha
from load_data_for_ML import load_all_data
from predict_SA_xgboost_covbat import predict_SA_xgboost_covbat
from predict_SA_ridge import predict_SA_ridge
from create_predictor_target_vars import create_predictor_target_vars
import pandas as pd

target = "BRIEF2_GEC_T_score"
metric = 'subcort_infant'
#options 'volume_infant', 'volume_VSA', 'subcort_VSA', 'subcort_infant', 'ad_VSA', 'rd_VSA', 'md_VSA', 'fa_VSA'
#        'surface_area_VSA', 'cortical_thickness_VSA', 'subcort_infant+volume_infant'
include_group = 0
bootstrap = 1
n_bootstraps = 100
show_heat_map = 0
remove_colinear = 0
run_dummy_quick_fit_xgb = 0
alpha=0.05
include_asd_in_train = 0

run_ridge_regression_fit = 0
run_xgboost_fit = 1
set_xgb_params_man = 0

if set_xgb_params_man:
    # Define parameter ranges to be used (ranges if BayesCV will be used)
    params = {"n_estimators": 50,  # (50, 2001),# Number of trees to create during training
              "min_child_weight": 1,
              # (1,11) # the number of samples required in each child node before attempting to split further
              "gamma": 0.01,
              # (0.01, 5.0, "log-uniform"),# regularization. Low values allow splits as long as they improve the loss function, no matter how small
              "eta": 0.01542,  # (0.005, 0.5, "log-uniform"),# learning rate
              "subsample": 0.576,  # (0.2, 1.0),# Fraction of training dta that is sampled for each boosting round
              "colsample_bytree": 0.2,  # (0.2, 1.0)  the fraction of features to be selected for each tree
              "max_depth": 6  # (2, 6), }#maximum depth of each decision tree
              }
else:
    # Define parameter ranges to be used (ranges if BayesCV will be used)
    params = {"n_estimators": (50, 2001),# Number of trees to create during training
              "min_child_weight": (1,11), # the number of samples required in each child node before attempting to split further
              "gamma": (0.01, 5.0, "log-uniform"),# regularization. Low values allow splits as long as they improve the loss function, no matter how small
              "eta": (0.005, 0.5, "log-uniform"),# learning rate
              "subsample": (0.2, 1.0),# Fraction of training dta that is sampled for each boosting round
              "colsample_bytree": (0.2, 1.0), # the fraction of features to be selected for each tree
              "max_depth": (2, 6) #maximum depth of each decision tree
              }

# Load and clean data for selected target and metric
df = load_all_data()

X, y, group_vals, sex_vals, X_test, y_test = (
    create_predictor_target_vars(df, target, metric, include_group, run_dummy_quick_fit_xgb,
                                    show_heat_map, remove_colinear, include_asd_in_train))

# # Plot histogram of target variable
# plt.hist(y, bins=30, edgecolor='black')
# plt.xlabel(target)
# plt.ylabel('Frequency')
# plt.title(f'Histogram of {target}')
# plt.grid(True)
# plt.show()

print(f"Running with target = {target} metric = {metric} include_asd_in_train= {include_asd_in_train} include_group = {include_group} "
      f"quick fit = {run_dummy_quick_fit_xgb}")

if run_xgboost_fit:
    y = pd.Series(y)
    y = pd.Series(y)
    # Use XGBoost to predict school age behavior from brain metric
    r2_val_xgb, feature_importance_df, r2_test_xgb = predict_SA_xgboost_covbat(X, y, target, metric, params,
                    run_dummy_quick_fit_xgb, X_test=X_test, y_test=y_test)

    # Calculate_xgb_percentile for r2test
    result_text_xgb, percentile_value_xgb = calculate_percentile(r2_val_xgb, alpha)

    # Plot distribution of r2 test
    plot_r2_distribution(r2_val_xgb, result_text_xgb, percentile_value_xgb, target, metric,
                         alpha, n_bootstraps, include_asd_in_train, alg='XGBoost')
    plt.show()

if run_ridge_regression_fit:
    # Get best alpha for ridge regression fit
    best_ridge_alpha = tune_ridge_alpha(X, y)

    # Run ridge regression
    r2_test_array_ridge, coef_summary_ridge = predict_SA_ridge(X, y, target, best_ridge_alpha, n_bootstraps)

    # Calculate ridge_percentile for r2test
    result_text_ridge, percentile_value_ridge = calculate_percentile(r2_test_array_ridge, alpha)

    # Plot distribution of r2 test
    plot_r2_distribution(r2_test_array_ridge, result_text_ridge, percentile_value_ridge, target, metric,
                         alpha, n_bootstraps, include_asd_in_train, alg='Ridge regression')
    plt.show()

    coef_summary_ridge.to_csv(f'Ridge_Regression_top_10_coefficients_{target}_{metric}_{n_bootstraps}.csv', index=False)