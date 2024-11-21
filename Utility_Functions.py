import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns


# Take DOB data frame and create a single data frame with one Sex, one Risk,a nd one Dob columns for all Identifiers

def simplify_dob_df(dob_risk_sex):
    df = dob_risk_sex.copy()
    # Replace '.' with NaN for easier handling
    df.replace('.', np.nan, inplace=True)
    # Find all columns that have DOB (there are ones for different ages)
    dob_cols = [col for col in df.columns if 'DoB' in col]
    dob_df = df[dob_cols].copy()
    # Collapse DoB values from all DoB columns into one column. Do the same for Risk and Sex
    dob_df_final = df['Identifiers'].to_frame().copy()
    dob_df_final['DoB'] = dob_df.apply(lambda row: row.dropna().iloc[0] if row.dropna().nunique() == 1 else pd.NA, axis=1)
    risk_cols = [col for col in df.columns if 'Risk' in col]
    tmp_df = df[risk_cols].copy()
    dob_df_final['Risk'] = tmp_df.apply(lambda row: row.dropna().iloc[0] if row.dropna().nunique() == 1 else pd.NA, axis=1)
    sex_cols = [col for col in df.columns if 'Sex' in col]
    tmp_df = df[sex_cols].copy()
    dob_df_final['Sex'] = tmp_df.apply(lambda row: row.dropna().iloc[0] if row.dropna().nunique() == 1 else pd.NA, axis=1)
    return dob_df_final

# Combine columns with 'VSA' and 'VSA-CVD'
def combine_vsa_columns(df):
    # Replace '.' with NaN so fillna() can handle it
    df = df.replace('.', np.nan)
    df = df.replace('#NULL!', np.nan)
    df = df.replace('nan', np.nan)
    df = df.replace('NaN', np.nan)
    df = df.replace('Unknown', np.nan)
    columns_to_drop = []
    # Iterate through all column names
    for col in df.columns:
        if 'VSA-CVD' in col:
            vsa_col = col.replace('VSA-CVD', 'VSA')
            if vsa_col in df.columns:
                new_col = col.replace('VSA-CVD', 'VSD-All')
                df[new_col] = df[vsa_col].fillna(df[col])
                # Mark the original columns for removal
                columns_to_drop.extend([vsa_col, col])
    df = df.drop(columns=columns_to_drop)
    return df

# For anotb, change all scores to NaN that have validitycode=3
# Columns are type object. Convert validity codes to integer type
def remove_invalid_anotb_data(anotb):
    anotb['V12_AB_validitycode'] = pd.to_numeric(anotb['V12_AB_validitycode'], errors='coerce')
    anotb['V24_AB_validitycode'] = pd.to_numeric(anotb['V24_AB_validitycode'], errors='coerce')
    anotb['V12_AB_validitycode'] = anotb['V12_AB_validitycode'].astype('Int64')
    anotb['V24_AB_validitycode'] = anotb['V24_AB_validitycode'].astype('Int64')
    # If validity code is 3, replace score with nan
    anotb.loc[anotb['V12_AB_validitycode'] == 3, 'AB_12_Percent'] = np.nan
    anotb.loc[anotb['V24_AB_validitycode'] == 3, 'AB_24_Percent'] = np.nan

    # Remove validity code columns
    anotb.drop(columns=['V12_AB_validitycode', 'V24_AB_validitycode'], inplace=True)
    return anotb

# Replace missing values with NaN
def replace_missing_with_nan(df):
    df.replace('#NULL!', np.nan, inplace=True)
    df.replace('.', np.nan, inplace=True)
    df.replace('nan', np.nan, inplace=True)
    df.replace('NaN', np.nan, inplace=True)
    df.replace('Unknown', np.nan, inplace=True)
    return df

# Remove all rows that have Down Syndrome Infant or Fragile X for VXX demographics,Project
def remove_fragx_downsyndrome_subj(df):
    proj_columns = [col for col in df.columns if 'Project' in col]
    IBIS_demograph_behavior_df = df[~df[proj_columns]
    .applymap(lambda cell: 'Fragile' in cell or 'Down' in cell if pd.notna(cell) else False)
    .any(axis=1)].copy()
    IBIS_demograph_behavior_df.drop(columns=proj_columns, inplace=True)
    return IBIS_demograph_behavior_df

# Remove extra text in columns that have ASD diagnostic category at later age than 24 months
def remove_24mo_extra_ASD_DX_text(df):
    df['V24 demographics,ASD_DX'] = np.where(df['V24 demographics,ASD_DX'].fillna('').str.contains('NO', regex=False), 'ASD-',
                 np.where(df['V24 demographics,ASD_DX'].fillna('').str.contains('YES', regex=False), 'ASD+',
                 np.nan))

    return df

# Remove extra text in columns that have ASD diagnostic category at 24 months
def remove_extra_ASD_DX_text(df):
    df['test'] = np.where(df['VSD-All demographics,ASD_Ever_DSMIV'].fillna('').str.contains('ASD+', regex=False), 'ASD+',
                 np.where(df['VSD-All demographics,ASD_Ever_DSMIV'].fillna('').str.contains('ASD-', regex=False), 'ASD-',
                 np.nan))

    test = df.pop('test')
    df.insert(2, 'test',test)
    df.rename(columns={'test': 'ASD_Ever_DSMIV', 'VSD-All NIHToolBox,Inst_24': 'NIHToolBox,TestName1',
                                               'VSD-All NIHToolBox,Inst_25': 'NIHToolBox,TestName2',
                                               'VSD-All NIHToolBox,Date_taken': 'NIHToolBox,Date_taken'}, inplace=True)
    return df

def remove_extra_text_eftasks(df):
    df['NIHToolBox,TestName1'] = np.where(
        df['NIHToolBox,TestName1'].fillna('').str.contains('Flanker', regex=False), 'Flanker',
        np.where(df['NIHToolBox,TestName1'].fillna('').str.contains('Dimensional', regex=False),
                 'DCCS',
                 np.nan))
    df['NIHToolBox,TestName2'] = np.where(
        df['NIHToolBox,TestName2'].fillna('').str.contains('Flanker', regex=False), 'Flanker',
        np.where(df['NIHToolBox,TestName2'].fillna('').str.contains('Dimensional', regex=False),
                 'DCCS',
                 np.nan))
    return df

def convert_numeric_columns_to_numeric_type(df):
    # Get list of columns with data type 'object'. Convert the ones that should be numeric to numeric type
    object_columns = df.select_dtypes(include='object').columns.tolist()
    categorical_columns = ['DoB', 'Identifiers', 'ASD_Ever_DSMIV', 'Risk', 'Sex', 'NIHToolBox,TestName1',
                           'NIHToolBox,TestName2',
                           'V06 demographics,Risk', 'V12 demographics,Risk', 'V24 demographics,Risk',
                           'V06 demographics,Sex',
                           'V12 demographics,Sex', 'V24 demographics,Sex', 'NIHToolBox,Date_taken',
                           'V06 demographics,Project',
                           'V12 demographics,Project', 'V24 demographics,Project', 'V12_AB_validitycode',
                           'V24_AB_validitycode', 'V24 demographics,ASD_DX', 'Combined_ASD_DX']
    cols_convert_to_numeric = [col for col in object_columns if col not in categorical_columns]
    df[cols_convert_to_numeric] = df[cols_convert_to_numeric].apply(
        pd.to_numeric, errors='coerce')
    return df

def remove_subj_no_behav_data(df):
    cols_demographic = ['Identifiers', 'Risk', 'Sex', 'DoB', 'ASD_Ever_DSMIV', 'V12prefrontal_taskCandidate_Age',
                        'V24prefrontal_taskCandidate_Age', 'V24 demographics,ASD_DX', 'Combined_ASD_DX']
    cols_not_demographic = [col for col in df.columns if col not in cols_demographic]
    ctest = df.loc[:, cols_not_demographic].copy()
    ctest = ctest.replace(['nan', 'NaN'], np.nan)
    df.replace(['nan', 'NaN'], np.nan, inplace=True)
    all_nan_rows = ctest[ctest.isna().all(axis=1)]
    all_nan_rows_IBIS = df[df.loc[:, cols_not_demographic].isna().all(axis=1)]
    ctest.dropna(how='all', inplace=True, ignore_index=True)
    df.dropna(subset=cols_not_demographic, how='all', inplace=True, ignore_index=True)
    return df

def make_flanker_dccs_columns(df):
    df['Flanker_Standard_Age_Corrected'] = df.apply(lambda row:
                                                    row['VSD-All NIHToolBox,Score_Age_Corrected_Standard_Score_1']
                                                    if row['NIHToolBox,TestName1'] == 'Flanker'
                                                    else (row['VSD-All NIHToolBox,Score_Age_Corrected_Standard_Score_2']
                                                    if row['NIHToolBox,TestName1'] == 'DCCS'
                                                    else np.nan),
                                                    axis=1)

    df['DCCS_Standard_Age_Corrected'] = df.apply(lambda row:
                                                    row['VSD-All NIHToolBox,Score_Age_Corrected_Standard_Score_1']
                                                    if row['NIHToolBox,TestName1'] == 'DCCS'
                                                    else (row['VSD-All NIHToolBox,Score_Age_Corrected_Standard_Score_2']
                                                    if row['NIHToolBox,TestName1'] == 'Flanker'
                                                    else np.nan),
                                                    axis=1)

    df.drop(columns=['VSD-All NIHToolBox,Score_Age_Corrected_Standard_Score_1',
                     'VSD-All NIHToolBox,Score_Age_Corrected_Standard_Score_2',
                     'NIHToolBox,TestName1', 'NIHToolBox,TestName2'], inplace=True)

    return df

def make_and_plot_missing_data_map(df, working_dir, filename, figsize):
    df_nan_nonnan = df.applymap(lambda x: 1 if pd.notna(x) else 0).astype(int)
    df_nan_nonnan.drop(columns=['Identifiers'], inplace=True)

    cmap = LinearSegmentedColormap.from_list('bwcmap', ['white', 'blue'])
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df_nan_nonnan, cmap=cmap, xticklabels=True, cbar=False, linewidths=0.5, ax=ax)
    plt.xticks(rotation=45, ha='right')
    plt.title('IBIS data set  white=missing  blue=available')
    plt.gcf().subplots_adjust(bottom=0.8)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f'{working_dir}/{filename}.png')
    plt.show(block=False)

def remove_Brief2_columns(df):
    brief2_cols = [col for col in df.columns if 'BRIEF2' in col]
    df.drop(columns=brief2_cols, inplace=True)
    return df

def write_missing_to_file(df, working_dir):
    rows_no_dob = df[df['DoB'].isna()]
    rows_no_asddx = df[df['ASD_Ever_DSMIV'].isna()]

    rows_no_dob['Identifiers'].to_csv(f'{working_dir}/IBIS_subjects_with_no_DOB.csv', index=False)
    rows_no_asddx['Identifiers'].to_csv(f'{working_dir}/IBIS_subjects_with_no_ASD.csv', index=False)

def calculate_nihtoolbox_age(df):
    date_df = pd.DataFrame()
    date_df['start_date'] = pd.to_datetime(df['DoB'], format='%m/%d/%y')
    date_df['end_date'] = pd.to_datetime(df['NIHToolBox,Date_taken'], format='%m/%d/%y')

    date_df['days_diff'] = (date_df['end_date'] - date_df['start_date']).dt.days

    date_df['diff_in_months'] = date_df['days_diff']/30.44
    date_df['diff_in_months'] = date_df['diff_in_months'].where(date_df['start_date'].notna()
                                 & date_df['end_date'].notna(), np.nan)

    df['Age_Taken_Calculated'] = date_df['diff_in_months'].copy()
    return df

def combine_asd_dx(df_orig):
    df = df_orig.copy()
    df = replace_missing_with_nan(df)
    df['Combined_ASD_DX'] = df['ASD_Ever_DSMIV'].fillna(df['V24 demographics,ASD_DX'])
    combined=df.pop('Combined_ASD_DX')
    df.insert(2, 'Combined_ASD_DX', combined)
    return df