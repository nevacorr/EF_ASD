
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

def load_subcortical_data(filepath):
    # Get all the CSV filenames in the directory
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
    merged_df = list(dfs_to_merge.values())[0]
    for df in list(dfs_to_merge.values())[1:]:
        merged_df = merged_df.merge(df, on='DCCID', how='outer')

    # Remove columns that contain the string "Edited"
    merged_df = merged_df.loc[:, ~merged_df.columns.str.contains('Edited')]

    # Divide all columns with '_v12' by 'totTiss_v12'
    v12_columns = [col for col in merged_df.columns if '_v12' in col and 'totTiss' not in col]
    for col in v12_columns:
        merged_df[col] = merged_df[col] / merged_df['totTiss_v12']
        mystop=1

    # Divide all columns with '_v24' by 'totTiss_v24'
    v24_columns = [col for col in merged_df.columns if '_v24' in col and 'totTiss' not in col]
    for col in v24_columns:
        merged_df[col] = merged_df[col] / merged_df['totTiss_v24']

    # Drop the 'totTiss_v12' and 'totTiss_v24' columns
    merged_df.drop(['totTiss_v12', 'totTiss_v24'], axis=1, inplace=True)

    mystop=1

