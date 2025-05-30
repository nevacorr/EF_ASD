import os
from load_brain_data import load_and_clean_dti_data, load_subcortical_data, load_and_clean_volume_data
from Utility_Functions_XGBoost import plot_correlations, remove_collinearity
from matplotlib import pyplot as plt
def load_all_data(target, metric, include_group_feature, run_dummy_quick_fit, show_heat_map, remove_colinear):

    print(f"Running with target = {target} metric = {metric} include_group = {include_group_feature} "
          f"quick fit = {run_dummy_quick_fit}")

    target = target
    metric = metric
    show_correlation_heatmap = show_heat_map
    remove_collinear_features = remove_colinear
    include_group_feature = include_group_feature

    # Define directories to be used
    working_dir = os.getcwd()
    vol_infant_dir = "/Users/nevao/R_Projects/IBIS_EF/"
    volume_infant_datafilename = "final_df_for_xgboost.csv"

    #############################
    #### Load VSA DTI data ######
    #############################
    dti_vsa_dir = ("/Users/nevao/Documents/IBIS_EF/source_data/Brain_Data/updated imaging_2-27-25/"
               "IBISandDS_VSA_DTI_Siemens_CMRR_v02.02_20250227/Siemens_CMRR/")
    ad_vsa_datafilename = "IBISandDS_VSA_DTI_SiemensAndCMRR_FiberAverage_AD_v02.02_20250227.csv"
    fa_vsa_datafilename = "IBISandDS_VSA_DTI_SiemensAndCMRR_FiberAverage_FA_v02.02_20250227.csv"
    md_vsa_datafilename = "IBISandDS_VSA_DTI_SiemensAndCMRR_FiberAverage_MD_v02.02_20250227.csv"
    rd_vsa_datafilename = "IBISandDS_VSA_DTI_SiemensAndCMRR_FiberAverage_RD_v02.02_20250227.csv"
    data_filenames = {
        "fa_VSA": fa_vsa_datafilename,"ad_VSA": ad_vsa_datafilename,"md_VSA": md_vsa_datafilename,"rd_VSA": rd_vsa_datafilename
    }
    datafilename = data_filenames.get(metric)
    df = load_and_clean_dti_data(dti_vsa_dir, datafilename, vol_infant_dir, volume_infant_datafilename, target, include_group_feature)

    #############################
    #### Load infant parcel volume data ######
    #############################
    datafilename = volume_infant_datafilename
    df = load_and_clean_volume_data(vol_infant_dir, datafilename, target, include_group_feature)

    #############################
    #### Load infant subcort volume data ######
    #############################
    datafilename = volume_infant_datafilename
    subcort_dir = '/Users/nevao/Documents/Genz/source_data/IBIS1&2_volumes_v3.13'
    df = load_subcortical_data(subcort_dir, vol_infant_dir, datafilename, target,  include_group_feature)
    df = df.reset_index(drop=True)

    return df
