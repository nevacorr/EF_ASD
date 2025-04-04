
import os
import pandas as pd
import re

def reshape_dataframe(df):
    df = df.drop(columns=["Combined_ID"])
    region_columns = [col for col in df.columns
                      if '_VQC' not in col and '_ExcludeReason' not in col and '_Edited' not in col
                      and col not in ["DCCID", "Visit"]]
    df_pivot = df.pivot(index='DCCID', columns="Visit", values=region_columns)
    df_pivot.columns = [f'{col[0]}_{col[1].lower()}' for col in df_pivot.columns]
    df_pivot.reset_index(inplace=True)

    return df_pivot

def load_subcortical_data(filepath, vol_dir, voldatafile, target, include_group):
    vol_df = pd.read_csv(f"{vol_dir}/{voldatafile}")

    vol_df = vol_df[vol_df['CandID'].notna()]

    vol_df['CandID'] = vol_df['CandID'].astype('int64')

    vol_df = vol_df.loc[:, ~vol_df.columns.str.contains('GM|WM|ICV|totTiss', regex=True)]

    vol_df.drop(columns=['Identifiers'], inplace=True)

    # Get all the suortical CSV filenames in the directory
    csv_files = [f for f in os.listdir(filepath) if f.endswith('.csv')]

    # Create an empty dictionary to store dataframes
    dfs = {}

    # Load each CSV file into a separate dataframe
    for file in csv_files:
        file_name = os.path.join(filepath, file)
        # Read each CSV into a DataFrame and store it in the dictionary
        # Extract the key using regex: Find the string between 'IBIS_v3.13_' and the next '_'
        match = re.search(r"IBIS_v3\.13_([^_]+)", file)
        if match:
            key = match.group(1)  # Extract the matched part
        else:
            key = file  # use full filename if pattern not found

        # Read CSV into a DataFrame and store it in the dictionary
        dfs[key] = pd.read_csv(file_name)

    dfs_transformed = {key: reshape_dataframe(df) for key, df in dfs.items()}

    # Exclude the 'ICV' dataframe from merging
    dfs_to_merge = {key: df for key, df in dfs_transformed.items() if key != "ICV"}

    # Merge all dataframes on 'Combined_ID' and 'DCCID'
    subcort_merged_df = list(dfs_to_merge.values())[0]
    for df in list(dfs_to_merge.values())[1:]:
        subcort_merged_df = subcort_merged_df.merge(df, on='DCCID', how='outer')

    # Remove columns that contain the string "Edited"
    subcort_merged_df = subcort_merged_df.loc[:, ~subcort_merged_df.columns.str.contains('Edited')]

    # Divide all columns with '_v12' by 'totTiss_v12'
    v12_columns = [col for col in subcort_merged_df.columns if '_v12' in col and 'totTiss' not in col]
    for col in v12_columns:
        subcort_merged_df[col] = subcort_merged_df[col] / subcort_merged_df['totTiss_v12']
        mystop=1

    # Divide all columns with '_v24' by 'totTiss_v24'
    v24_columns = [col for col in subcort_merged_df.columns if '_v24' in col and 'totTiss' not in col]
    for col in v24_columns:
        subcort_merged_df[col] = subcort_merged_df[col] / subcort_merged_df['totTiss_v24']

    # Drop the 'totTiss_v12' and 'totTiss_v24' columns
    subcort_merged_df.drop(['totTiss_v12', 'totTiss_v24'], axis=1, inplace=True)

    subcort_merged_df.rename(columns={'DCCID': 'CandID'}, inplace=True)

    merged_df = vol_df.merge(subcort_merged_df, on='CandID', how='left')

    merged_df.drop(columns=['CandID'], inplace=True)

    # Keep only rows where the response variable is not NA
    merged_df = merged_df[merged_df[target].notna()]

    # Encode Sex column Female = 0 Male = 1
    merged_df["Sex"] = merged_df["Sex"].replace({"Female": 0, "Male": 1})

    if include_group:
        columns_to_exclude = ["Combined_ASD_DX", "Risk", "AB_12_Percent", "AB_24_Percent",
                              "BRIEF2_GEC_raw_score", "BRIEF2_GEC_T_score", "DCCS_Standard_Age_Corrected"]
    else:
        columns_to_exclude = ["Combined_ASD_DX", "Group", "Risk", "AB_12_Percent", "AB_24_Percent",
                              "BRIEF2_GEC_raw_score", "BRIEF2_GEC_T_score", "DCCS_Standard_Age_Corrected"]

    merged_df.drop(columns=columns_to_exclude, inplace=True)

    if include_group:
        # One-hot encode the 'Group' column
        merged_df = pd.get_dummies(merged_df, columns=['Group'], drop_first=False)

    merged_df = merged_df.dropna(subset=[col for col in merged_df.columns if col not in ['Sex', target]], how='all')

    return merged_df

def load_and_clean_dti_data(dir, datafilename, vol_dir, voldatafile, target, include_group):

    dti_df = pd.read_csv(f"{dir}/{datafilename}")

    vol_df = pd.read_csv(f"{vol_dir}/{voldatafile}")

    vol_df = vol_df[vol_df['CandID'].notna()]

    vol_df['CandID'] = vol_df['CandID'].astype('int64')

    vol_df = vol_df.loc[:, ~vol_df.columns.str.contains('GM|WM|ICV|totTiss', regex=True)]

    vol_df.drop(columns=['Identifiers'], inplace=True)

    dti_df.drop(columns=['Visit_label', 'CandID_Visit', 'dMRI_protocol', 'FileID'], inplace=True)

    dti_df = dti_df.loc[:, ~dti_df.columns.str.contains('Optic|Motor|Fornix|CorticoSpinal|UNC|Reticular|ILF|CT|CF|CC|Temporo|IFOF', regex=True)]

    dti_df.replace({'.': np.nan, '': np.nan}, inplace=True)

    dti_df[dti_df.columns.difference(['CandID'])] = (
        dti_df[dti_df.columns.difference(['CandID'])].apply(pd.to_numeric, errors='coerce'))

    merged_df = vol_df.merge(dti_df, on="CandID", how="left")

    merged_df.drop(columns=['CandID'], inplace=True)

    # Keep only rows where the response variable is not NA
    merged_df = merged_df[merged_df[target].notna()]

    # Encode Sex column Female = 0 Male = 1
    merged_df["Sex"] = merged_df["Sex"].replace({"Female": 0, "Male": 1})

    if include_group:
        columns_to_exclude = ["Combined_ASD_DX", "Risk", "AB_12_Percent", "AB_24_Percent",
                              "BRIEF2_GEC_raw_score", "BRIEF2_GEC_T_score", "DCCS_Standard_Age_Corrected"]
    else:
        columns_to_exclude = ["Combined_ASD_DX","Group", "Risk", "AB_12_Percent", "AB_24_Percent",
                              "BRIEF2_GEC_raw_score", "BRIEF2_GEC_T_score", "DCCS_Standard_Age_Corrected"]

    merged_df.drop(columns=columns_to_exclude, inplace=True)

    if include_group:
        # One-hot encode the 'Group' column
        merged_df = pd.get_dummies(merged_df, columns=['Group'], drop_first=False)

    merged_df = merged_df.dropna(subset=[col for col in merged_df.columns if col not in ['Sex', target]], how='all')

    return merged_df


def load_and_clean_volume_data(filepath, filename, target, include_group):

    df = pd.read_csv(f"{filepath}/{filename}")

    if include_group:
        columns_to_exclude = ["CandID", "Identifiers", "Combined_ASD_DX", "Risk", "AB_12_Percent", "AB_24_Percent",
                              "BRIEF2_GEC_raw_score", "BRIEF2_GEC_T_score", "DCCS_Standard_Age_Corrected",
                              "Flanker_Standard_Age_Corrected", "ICV_V12","ICV_V24", "totTiss_V12", "totTiss_V24"]

        columns_to_exclude.remove(target)

    else:
        columns_to_exclude = ["CandID", "Group", "Identifiers", "Combined_ASD_DX", "Risk", "AB_12_Percent", "AB_24_Percent",
                              "BRIEF2_GEC_raw_score", "BRIEF2_GEC_T_score", "DCCS_Standard_Age_Corrected",
                              "Flanker_Standard_Age_Corrected","ICV_V12","ICV_V24", "totTiss_V12", "totTiss_V24"]

        columns_to_exclude.remove(target)

    df.drop(columns=columns_to_exclude, inplace=True)

    # Keep only rows where the response variable is not NA
    df = df[df[target].notna()]

    # Encode Sex column Female = 0 Male = 1
    df["Sex"] = df["Sex"].replace({"Female": 0, "Male": 1})

    if include_group:
        # One-hot encode the 'Group' column
        df = pd.get_dummies(df, columns=['Group'], drop_first=False)

    return df


