import os
from load_brain_data import load_and_clean_dti_data, load_subcortical_data, load_and_clean_volume_data
from Utility_Functions_XGBoost import plot_correlations, remove_collinearity
from matplotlib import pyplot as plt
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
