from load_data_for_ML import load_all_data
import itertools
from predict_SA_xgboost import predict_SA_xgboost
from create_predictor_target_vars import create_predictor_target_vars

show_heat_map = 0
remove_colinear = 0
run_dummy_quick_fit_xgb = 0
set_params_man = 0

targets = ["Flanker_Standard_Age_Corrected", "BRIEF2_GEC_raw_score","BRIEF2_GEC_T_score", "DCCS_Standard_Age_Corrected"]
metrics = [ 'cortical_thickness_VSA', 'volume_VSA','subcort_VSA', 'surface_area_VSA', 'volume_infant', 'subcort_infant',
            'fa_VSA', 'md_VSA', 'rd_VSA', 'ad_VSA']
include_group_options = [0]

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

df = load_all_data()

for target, metric, include_group in itertools.product(targets, metrics, include_group_options):

    X, y = create_predictor_target_vars(df, target, metric, include_group, run_dummy_quick_fit_xgb,
                                        show_heat_map, remove_colinear)

    print(f"Running with target = {target} metric = {metric} include_group = {include_group} "
          f"quick fit = {run_dummy_quick_fit_xgb}")

    predict_SA_xgboost(
        X,
        y,
        target,
        metric,
        params,
        run_dummy_quick_fit_xgb,
        set_params_man,
        show_results_plot=0,
        bootstrap = 0,
        n_bootstraps=None)