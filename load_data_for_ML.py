import os
import pandas as pd
from load_brain_data import load_and_clean_vsa_dti_data, load_infant_subcortical_data
from load_brain_data import load_and_clean_infant_volume_data_and_all_behavior
from load_brain_data import load_and_clean_vsa_volume_data, load_vsa_subcortical_data
from load_brain_data import load_vsa_ct_data
from functools import reduce
from Utility_Functions import divide_columns_by_tottiss
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
    df_infant_dem_lobe = load_and_clean_infant_volume_data_and_all_behavior(vol_infant_dir, datafilename)

    #############################
    #### Load school age lobe volume data and age data ######
    #############################
    vol_dir_SA = "/Users/nevao/Documents/IBIS_EF/source data/Brain_Data/updated imaging_2-27-25/IBISandDS_VSA_Cerebrum_and_LobeParcel_Vols_v01.04_20250221"
    volume_SA_datafilename = 'IBISandDS-VSA_Lobe_Vols_v01.04_20250221.csv'
    tot_tiss_dir_SA = "/Users/nevao/Documents/IBIS_EF/source data/Brain_Data/updated imaging_2-27-25/IBISandDS_VSA_TissueSeg_Vols_v01.04_20250221"
    tot_tiss_SA_datafilename = 'IBISandDS_VSA_TissueSeg_Vols_v01.04_20250221.csv'
    df_vsa_lobe = load_and_clean_vsa_volume_data(vol_dir_SA, volume_SA_datafilename,
                                                 tot_tiss_dir_SA, tot_tiss_SA_datafilename)

    #############################
    #### Load infant subcort volume data ######
    #############################
    subcort_infant_dir = '/Users/nevao/Documents/IBIS_EF/source data/Brain_Data/IBIS1&2_volumes_v3.13'
    df = load_infant_subcortical_data(subcort_infant_dir)
    df_infant_subcort = df.reset_index(drop=True)

    #############################
    #### Load school age subcort volume data ######
    #############################
    subcort_vsa_dir = "/Users/nevao/Documents/IBIS_EF/source data/Brain_Data/updated imaging_2-27-25/IBISandDS_VSA_Subcort_and_LV_Vols_v01.04_20250221"
    subcort_vsa_datafilename = 'IBISandDS_VSA_Subcort_and_LV_Vols_v01.04_20250221.csv'
    df_vsa_subcort = load_vsa_subcortical_data(subcort_vsa_dir, subcort_vsa_datafilename)

    #############################
    #### Load school age cortical thickness data ######
    #############################
    ct_vsa_dir = "/Users/nevao/Documents/IBIS_EF/source data/Brain_Data/IBISandDS_VSA_SurfaceData_v01.02_20210809"
    ct_vsa_datafilename = 'IBISandDS_VSA_CorticalThickness_DKT_v01.02_20210708.csv'
    df_vsa_ct = load_vsa_ct_data(ct_vsa_dir, ct_vsa_datafilename)

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
    df_vsa_dti = pd.concat(dfs, axis=1)

    #############################
    #### Combine all data ######
    #############################

    dfs_list= [df_infant_dem_lobe, df_vsa_lobe, df_infant_subcort, df_vsa_subcort, df_vsa_ct, df_vsa_dti]

    dfs_combined = reduce(lambda left, right: pd.merge(left, right, on='CandID', how='outer'), dfs_list)

    dfs_combined['CandID'] = pd.to_numeric(dfs_combined['CandID'], errors="coerce").astype("Int64")

    # remove rows that have Nan values for all brain measures
    non_brain_cols = [
        "CandID", "Identifiers", "Combined_ASD_DX", "Risk", "Sex",
        "AB_12_Percent", "AB_24_Percent",
        "BRIEF2_GEC_T_score", "BRIEF2_GEC_raw_score",
        "Flanker_Standard_Age_Corrected", "DCCS_Standard_Age_Corrected",
        "Group_HR+", "Group_HR-", "Group_LR-"
    ]

    cols_to_check = dfs_combined.columns.difference(non_brain_cols)

    dfs_all = dfs_combined[~dfs_combined[cols_to_check].isna().all(axis=1)]

    # Determine which rows had no brain data and were removed
    diff = dfs_combined.merge(dfs_all, how='outer', indicator=True)
    rows_only_in_df1 = diff[diff['_merge'] == 'left_only'].drop(columns=['_merge'])

    # remove rows that have nan values for all behavior measures
    behav_cols = [
        "AB_12_Percent", "AB_24_Percent",
        "BRIEF2_GEC_T_score", "BRIEF2_GEC_raw_score",
        "Flanker_Standard_Age_Corrected", "DCCS_Standard_Age_Corrected",
    ]

    df_all_brain_behav = dfs_all.dropna(subset=behav_cols, how='all')

    # Divide call volume columns by totTissue for the appropriate age
    df_all_brain_behav = divide_columns_by_tottiss(df_all_brain_behav, df_infant_dem_lobe, "V12")
    df_all_brain_behav = divide_columns_by_tottiss(df_all_brain_behav, df_infant_dem_lobe, "V24")
    df_all_brain_behav = divide_columns_by_tottiss(df_all_brain_behav, df_vsa_lobe, "VSA")
    df_all_brain_behav = divide_columns_by_tottiss(df_all_brain_behav, df_infant_subcort, "v12")
    df_all_brain_behav = divide_columns_by_tottiss(df_all_brain_behav, df_infant_subcort, "v24")
    df_all_brain_behav = divide_columns_by_tottiss(df_all_brain_behav, df_vsa_subcort, "VSA")

    df_all_brain_behav = df_all_brain_behav.loc[:, ~df_all_brain_behav.columns.str.contains('Tiss|ICV')]

    return df_all_brain_behav
