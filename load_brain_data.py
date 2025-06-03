
import os
import pandas as pd
import re
import numpy as np

def reshape_dataframe(df):
    df = df.drop(columns=["Combined_ID"])
    region_columns = [col for col in df.columns
                      if '_VQC' not in col and '_ExcludeReason' not in col and 'Edited' not in col
                      and col not in ["DCCID", "Visit"]]
    df_pivot = df.pivot(index='DCCID', columns="Visit", values=region_columns)
    df_pivot.columns = [f'{col[0]}_{col[1].lower()}' for col in df_pivot.columns]
    df_pivot.reset_index(inplace=True)

    return df_pivot

def load_infant_subcortical_data(filepath):

    # Get all the subcortical CSV filenames in the directory
    csv_files = [f for f in os.listdir(filepath) if f.endswith('.csv') and 'LobeParcel' not in f]

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

    return subcort_merged_df

def load_and_clean_vsa_dti_data(dir, datafilename):

    dti_df = pd.read_csv(f"{dir}/{datafilename}")

    dti_df.drop(columns=['Visit_label', 'CandID_Visit', 'dMRI_protocol', 'FileID'], inplace=True)

    dti_df = dti_df.loc[:, ~dti_df.columns.str.contains('Optic|Motor|Fornix|CorticoSpinal|UNC|Reticular|ILF|CT|CF|CC|Temporo|IFOF', regex=True)]

    dti_df.replace({'.': np.nan, '': np.nan}, inplace=True)

    dti_df[dti_df.columns.difference(['CandID'])] = (
        dti_df[dti_df.columns.difference(['CandID'])].apply(pd.to_numeric, errors='coerce'))

    return dti_df


def load_and_clean_infant_volume_data_and_all_behavior(filepath, filename):

    df = pd.read_csv(f"{filepath}/{filename}")

    # Encode Sex column Female = 0 Male = 1
    df["Sex"] = df["Sex"].replace({"Female": 0, "Male": 1})

    # One-hot encode the 'Group' column
    df = pd.get_dummies(df, columns=['Group'], drop_first=False)

    # Identify the columns that start with "Group"
    group_cols = [col for col in df.columns if col.startswith("Group")]

    # Get the list of other columns (excluding group_cols)
    other_cols = [col for col in df.columns if col not in group_cols]

    #Insert group_cols into the desired positions (5th to 7th )
    new_col_order = other_cols[:5] + group_cols + other_cols[5:]

    # Step 4: Reorder the DataFrame
    df = df[new_col_order]

    return df


