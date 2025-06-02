import os
import pandas as pd
from load_brain_data import load_and_clean_vsa_dti_data, load_infant_subcortical_data
from load_brain_data import load_and_clean_infant_volume_data_and_all_behavior
from Utility_Functions_XGBoost import plot_correlations, remove_collinearity
from matplotlib import pyplot as plt
def load_all_data():

    # Define directories to be used
    working_dir = os.getcwd()
    vol_infant_dir = "/Users/nevao/R_Projects/IBIS_EF/"
    volume_infant_datafilename = "final_df_for_xgboost.csv"

    #############################
    #### Load infant lobe volume data and all age behavioral data ######
    #############################
    datafilename = volume_infant_datafilename
    df = load_and_clean_infant_volume_data_and_all_behavior(vol_infant_dir, datafilename)

    #############################
    #### Load infant subcort volume data ######
    #############################
    subcort_dir = '/Users/nevao/Documents/IBIS_EF/source data/Brain_Data/IBIS1&2_volumes_v3.13'
    df = load_infant_subcortical_data(subcort_dir)
    df = df.reset_index(drop=True)

    #############################
    #### Load VSA DTI data ######
    #############################
    dti_vsa_dir = ("/Users/nevao/Documents/IBIS_EF/source data/Brain_Data/updated imaging_2-27-25/"
               "IBISandDS_VSA_DTI_Siemens_CMRR_v02.02_20250227/Siemens_CMRR")

    metric_files = {
        "FA": "IBISandDS_VSA_DTI_SiemensAndCMRR_FiberAverage_AD_v02.02_20250227.csv",
        "AD": "IBISandDS_VSA_DTI_SiemensAndCMRR_FiberAverage_FA_v02.02_20250227.csv",
        "MD": "IBISandDS_VSA_DTI_SiemensAndCMRR_FiberAverage_MD_v02.02_20250227.csv",
        "RD": "IBISandDS_VSA_DTI_SiemensAndCMRR_FiberAverage_RD_v02.02_20250227.csv"
    }

    dfs = []
    for metric, filename in metric_files.items():
        df = load_and_clean_vsa_dti_data(dti_vsa_dir, filename)
        df.rename(columns={col: f"{metric}_{col}" for col in df.columns if col != 'CandID'}, inplace=True)
        dfs.append(df)

    # Drop duplicate CandID columns from all but the first dataframe
    dfs = [dfs[0]] + [df.drop(columns='CandID', errors='ignore') for df in dfs[1:]]

    # Concatenate all DataFrames column-wise
    df_concat = pd.concat(dfs, axis=1)

    return df_concat
