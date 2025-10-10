import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns

def make_lists_of_columns_needed(anotb, brief2, dx, dx2, nihtoolbox, asd_diagnosis, time1_demographics):

    ####### ---- A not B-----#######
    anotb_cols_to_keep = []
    anotb_cols_to_remove = list(anotb.columns.difference(anotb_cols_to_keep))

    ####### ---- BRIEF2-----#######
    brief2_cols_to_keep = []

    remove_list = ['VSA BRIEF2_Parent,T_score', 'VSA-CVD BRIEF2_Parent,T_score','VSA BRIEF2_Parent,raw_score',
                   'VSA-CVD BRIEF2_Parent,raw_score']
    brief2_cols_to_keep = [col for col in brief2_cols_to_keep if col not in remove_list]

    ####### ---- Brief1-----#######
    brief1_cols_to_keep = []

    ####### ---- Dx -----#######
    dx_col_to_keep = ['Identifiers']
    dx_col_to_keep.extend([col for col in dx.columns if 'ASD_Ever_DSMIV' in col])

    ####### ---- Dx2 -----#######
    # Make list of dx2 columns to keep
    dx2_substrings_to_keep = ['ethnicity', 'race']
    dx2_cols_to_keep = ['Identifiers']
    ####### Keep column if any of ths substrings is in each column name
    dx2_cols_to_keep.extend([col for col in dx2.columns if any(sub in col for sub in dx2_substrings_to_keep)])

    ####### ---- NIH Toolbox -----#######
    # Make a list of nih toolbox columns to keep
    nihtoolbox_cols_to_keep = ['Identifiers']
    nihtoolbox_substrings_to_keep = ['Date_taken', 'Ethnicity', 'Education', 'Race',
                                     'Candidate_Age']
    nihtoolbox_cols_to_keep.extend(
        [col for col in nihtoolbox.columns if any(sub in col for sub in nihtoolbox_substrings_to_keep)])

    ####### ---- ASD Diagnosis ---- #######
    asd_cols_to_keep = ['Identifiers']
    asd_cols_to_keep.extend([col for col in asd_diagnosis if 'Ever_DSMIV' in col])

    ########------ Time 1 Demographics -------#########
    demot1_substrings_to_keep = ['ethnicity', 'race', 'education']
    demot1_cols_to_keep = ['Identifiers']
    demot1_cols_to_keep.extend([col for col in time1_demographics.columns if any(sub in col for sub in demot1_substrings_to_keep)])
    return (anotb_cols_to_remove, brief2_cols_to_keep, brief1_cols_to_keep, dx_col_to_keep, dx2_cols_to_keep,
            nihtoolbox_cols_to_keep, asd_cols_to_keep, demot1_cols_to_keep)

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

# Remove extra text in columns that have ASD diagnostic at 24 months
def remove_24mo_extra_ASD_DX_text(df):
    df['V24 demographics,ASD_DX'] = np.where(df['V24 demographics,ASD_DX'].fillna('').str.contains('NO', regex=False), 'ASD-',
                 np.where(df['V24 demographics,ASD_DX'].fillna('').str.contains('YES', regex=False), 'ASD+',
                 np.nan))

    return df

# Remove extra text in columns that have ASD diagnostic category in Ever column
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

def remove_extra_text_asd_diagnosis(df):
    Identifiers = df['Identifiers'].copy()

    for col in df.columns:
        df[col] = np.where(df[col].fillna('').str.contains('ASD+', regex=False), 'ASD+',
               np.where(df[col].fillna('').str.contains('ASD-', regex=False), 'ASD-',
               np.nan))
    df['Identifiers'] = Identifiers
    df.replace('nan', np.nan, inplace=True)
    df['ASD_Ever_DSMIV_infant'] = (
         df['VSD-All demographics,ASD_Ever_DSMIV']
        .fillna(df['V37Plus demographics,ASD_Ever_DSMIV'])
        .fillna(df['V36 demographics,ASD_Ever_DSMIV'])
        .fillna(df['V24 demographics,ASD_Ever_DSMIV'])
    )
    df.drop(columns=['VSD-All demographics,ASD_Ever_DSMIV','V37Plus demographics,ASD_Ever_DSMIV',
                     'V36 demographics,ASD_Ever_DSMIV', 'V24 demographics,ASD_Ever_DSMIV'], inplace=True)
    mystop=1
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
                           'V24_AB_validitycode', 'V24 demographics,ASD_DX', 'Combined_ASD_DX',
                           'ASD_Ever_DSMIV_infant']
    cols_convert_to_numeric = [col for col in object_columns if col not in categorical_columns]
    df[cols_convert_to_numeric] = df[cols_convert_to_numeric].apply(
        pd.to_numeric, errors='coerce')
    return df

def remove_subj_no_behav_data(df):
    cols_demographic = ['Identifiers', 'Risk', 'Sex', 'DoB', 'ASD_Ever_DSMIV', 'V12prefrontal_taskCandidate_Age',
                        'V24prefrontal_taskCandidate_Age', 'V24 demographics,ASD_DX', 'Combined_ASD_DX',
                        'ASD_Ever_DSMIV_infant', 'NIHToolBox,Date_taken']
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
    brief2_cols.remove("VSD-All BRIEF2_Parent,GEC_T_score")
    brief2_cols.remove("VSD-All BRIEF2_Parent,GEC_raw_score")
    df.drop(columns=brief2_cols, inplace=True)
    return df

def write_missing_to_file(df, working_dir):
    rows_no_dob = df[df['DoB'].isna()]
    rows_no_asddx = df[df['Combined_ASD_DX'].isna()]

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
    df['Combined_ASD_DX'] = df['ASD_Ever_DSMIV'].fillna(df['ASD_Ever_DSMIV_infant'])
    combined=df.pop('Combined_ASD_DX')
    df.insert(2, 'Combined_ASD_DX', combined)
    df.drop(columns=['ASD_Ever_DSMIV', 'ASD_Ever_DSMIV_infant'], inplace=True)
    return df

def combine_age_nihtoolbox(df_orig):
    df = df_orig.copy()
    df['Age_SchoolAge'] = df['Age_Taken_Calculated'].fillna(df['VSD-All NIHToolBox,Candidate_Age'])
    df.drop(columns=['Age_Taken_Calculated', 'VSD-All NIHToolBox,Candidate_Age', 'NIHToolBox,Date_taken', 'DoB'],inplace=True)
    return df

def create_combined_dx_risk_column(df):
    # Standardize missing values
    df = df.replace({pd.NA: np.nan, None: np.nan})

    # Ensure consistent types
    df['Risk'] = df['Risk'].astype('object')
    df['Combined_ASD_DX'] = df['Combined_ASD_DX'].astype('object')

    # Fill missing values with default
    df['Risk'] = df['Risk'].fillna('Unknown')
    df['Combined_ASD_DX'] = df['Combined_ASD_DX'].fillna('Unknown')

    # Define conditions with explicit handling for 'Unknown'
    conditions = [
        (df['Risk'] == 'HR') & (df['Combined_ASD_DX'] == 'ASD+'),
        (df['Risk'] == 'HR') & (df['Combined_ASD_DX'] == 'ASD-'),
        (df['Risk'] == 'LR') & (df['Combined_ASD_DX'] == 'ASD+'),
        (df['Risk'] == 'LR') & (df['Combined_ASD_DX'] == 'ASD-')
    ]
    choices = ['HR+', 'HR-', 'LR+', 'LR-']

    # Use np.select to create the 'Group' column
    df['Group'] = np.select(conditions, choices, default=np.nan)

    group_col = df.pop('Group')
    df.insert(3, 'Group', group_col)

    df = replace_missing_with_nan(df)

    return df

def output_rows_with_nans(df, measure, columns_to_check, exclude_prefixes=None):
    # Filter rows where any of the specified columns are NaN
    mask = df[columns_to_check].isna().any(axis=1)
    filtered_df = df.loc[mask]

    # Drop columns with specified prefixes
    if exclude_prefixes is not None:
        cols_to_drop = [col for col in filtered_df.columns
                        if any(col.startswith(pref) for pref in exclude_prefixes)]
        filtered_df = filtered_df.drop(columns=cols_to_drop, errors='ignore')

    filtered_df.reset_index(inplace=True, drop=True)

    filtered_df.to_csv(f'/Users/nevao/PycharmProjects/EF_ASD/{measure}_with_missing_v24_and_age_DAS.csv', index=False)

def summarize_by_group(df):
    group_col = 'Group'

    # Ensure group column is string
    # Make a copy of dataframe and filter unwanted groups
    df_summary = df.copy()
    df_summary = df_summary[df_summary['Group'] != 'LR+']
    df_summary = df_summary.dropna(subset=['Group'])
    df_summary['Group'] = df_summary['Group'].astype(str)

    group_col = 'Group'

    # Fill missing values and standardize strings
    df_summary['V06.tsi.mother_education'] = df_summary['V06.tsi.mother_education'].fillna(
        'missing').str.strip().str.lower()
    df_summary['race'] = df_summary['race'].fillna('unknown_not_reported').str.strip().str.lower()
    df_summary['Sex'] = df_summary['Sex'].fillna('unknown').str.strip().str.capitalize()

    # 1. Total N per group
    total_n = df_summary[group_col].value_counts().rename("Total N").to_frame().T

    # 2. Sex (Male) n (%)
    sex_counts = pd.crosstab(df_summary[group_col], df_summary['Sex'])
    male_counts = sex_counts.get('Male', pd.Series(0, index=sex_counts.index))
    male_percent = (male_counts / sex_counts.sum(axis=1) * 100).round(1)
    sex_summary = (male_counts.astype(int).astype(str) + " (" + male_percent.astype(str) + "%)").to_frame().T
    sex_summary.index = ["Sex (Male)"]

    # 3. Maternal Education (auto-detect categories)
    edu_order = sorted(df_summary['V06.tsi.mother_education'].unique())
    edu_counts = pd.crosstab(df_summary['V06.tsi.mother_education'], df_summary[group_col]).reindex(edu_order,
                                                                                                    fill_value=0)

    # 4. Race (auto-detect categories)
    race_order = sorted(df_summary['race'].unique())
    race_counts = pd.crosstab(df_summary['race'], df_summary[group_col]).reindex(race_order, fill_value=0)

    # 5. Combine into one table with blank rows as separators
    blank_row = pd.DataFrame([[""] * len(total_n.columns)], columns=total_n.columns)
    table = pd.concat([
        total_n,
        sex_summary,
        blank_row,
        edu_counts,
        blank_row,
        race_counts
    ])

    # Optional: display
    print(table)

    return table


def compute_stats_conditioned_on_identifiers(df, categorical_columns=None):
    # This function computes descriptive statistics for each column in the dataframe,
    # conditioned on the presence of different identifier columns.
    #
    # Specifically:
    # - It looks for all columns whose names start with 'Identifiers_' and extracts their suffixes.
    # - For each suffix (e.g., 'ibis', 'vsa'), it selects the subset of rows where that identifier is not NaN.
    # - For every other column (excluding all 'Identifiers' columns):
    #     * If the column is categorical (either explicitly listed in categorical_columns or inferred as non-numeric),
    #       it records value counts (including NaNs) and the number of missing values.
    #     * If the column is numeric, it records mean, standard deviation, non-missing count, and number of missing values.
    # - The results are stored in a dictionary keyed as "<column>__<suffix>" and returned as a DataFrame.
    #
    # Example:
    # If you have identifiers 'Identifiers_ibis' and 'Identifiers_vsa',
    # this function will compute stats for each non-identifier column separately
    # within the subset of rows where 'Identifiers_ibis' is not NaN
    # and where 'Identifiers_vsa' is not NaN.
    #
    # Example:
    # Input dataframe:
    #    Identifiers_ab12  Identifiers_24  Identifiers_brief2  Identifiers_dccs  Identifiers_flanker   Age   Sex
    # 0             101.0            NaN               301.0             NaN                 NaN   10.0     M
    # 1             102.0            NaN                 NaN           401.0                 NaN   12.0     F
    # 2               NaN          201.0                 NaN             NaN               501.0   11.0     F
    # 3               NaN          202.0                 NaN             NaN                 NaN    NaN     M
    #
    # If categorical_columns = ['Sex'], then:
    #
    # - For suffix "ab12" (rows 0,1):
    #     Age__ab12 → mean=11.0, std=1.41, count=2, n_nan=0
    #     Sex__ab12 → value_counts={'M':1, 'F':1}, n_nan=0
    #
    # - For suffix "24" (rows 2,3):
    #     Age__24 → mean=11.0, std=0.0, count=1, n_nan=1
    #     Sex__24 → value_counts={'F':1, 'M':1}, n_nan=0
    #
    # - For suffix "brief2" (rows 0):
    #     Age__brief2 → mean=10.0, std=NaN, count=1, n_nan=0
    #     Sex__brief2 → value_counts={'M':1}, n_nan=0
    #
    # - For suffix "dccs" (rows 1):
    #     Age__dccs → mean=12.0, std=NaN, count=1, n_nan=0
    #     Sex__dccs → value_counts={'F':1}, n_nan=0
    #
    # - For suffix "flanker" (rows 2):
    #     Age__flanker → mean=11.0, std=NaN, count=1, n_nan=0
    #     Sex__flanker → value_counts={'F':1}, n_nan=0
    #
    # The returned DataFrame would look like:
    #
    #                          mean   std  count  n_nan                     value_counts
    # Age__ab12                11.0  1.41    2.0      0                            NaN
    # Sex__ab12                  NaN   NaN    NaN      0        {'M':1, 'F':1}
    # Age__24                   11.0  0.0     1.0      1                            NaN
    # Sex__24                    NaN   NaN    NaN      0        {'F':1, 'M':1}
    # Age__brief2               10.0  NaN     1.0      0                            NaN
    # Sex__brief2                NaN   NaN    NaN      0        {'M':1}
    # Age__dccs                 12.0  NaN     1.0      0                            NaN
    # Sex__dccs                  NaN   NaN    NaN      0        {'F':1}
    # Age__flanker              11.0  NaN     1.0      0                            NaN
    # Sex__flanker               NaN   NaN    NaN      0        {'F':1}

    if categorical_columns is None:
        categorical_columns = []

    result = {}

    # Identify columns that start with 'Identifiers_' and store suffixes
    identifier_cols = [col for col in df.columns if col.startswith('Identifiers_')]
    suffixes = [
        col.removeprefix("Identifiers_").removesuffix("__noIQ_ME")
        for col in identifier_cols
    ]
    for suffix in suffixes:
        id_col = f'Identifiers_{suffix}__noIQ_ME'
        sub_df = df[df[id_col].notna()]  # Only keep rows where this identifier is not NaN

        sub_df.to_csv(f'Demographics for {suffix}.csv')

        # Loop through other columns, excluding all Identifiers columns
        for col in sub_df.columns:
            if col.startswith('Identifiers') or col == 'Identifiers':
                continue

            values = sub_df[col]

            # Treat explicitly listed categorical columns as non-numeric
            if col in categorical_columns or not pd.api.types.is_numeric_dtype(values):
                value_counts = values.value_counts(dropna=False).to_dict()
                stats = {
                    'value_counts': value_counts,
                    'n_nan': values.isna().sum()
                }
            else:
                stats = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'count': values.count(),
                    'n_nan': values.isna().sum()
                }

            result[f"{col}__{suffix}"] = stats

    return pd.DataFrame(result).T

def compute_stats_conditioned_on_identifiers_by_group(df, categorical_columns=None, group_col="Group"):

    if categorical_columns is None:
        categorical_columns = []

    result = {}

    # Identify columns that start with 'Identifiers_' and store suffixes
    identifier_cols = [col for col in df.columns if col.startswith('Identifiers_')]
    suffixes = [
        col.removeprefix("Identifiers_").removesuffix("__noIQ_ME")
        for col in identifier_cols
    ]
    for suffix in suffixes:
        id_col = f'Identifiers_{suffix}__noIQ_ME'
        sub_df = df[df[id_col].notna()]  # Only keep rows where this identifier is not NaN

        # Now split by group within this identifier
        for group, group_df in sub_df.groupby(group_col):
            group_key = f"{suffix}__{group}"

            # Loop through other columns, excluding all Identifiers columns
            for col in group_df.columns:
                if col.startswith('Identifiers') or col == 'Identifiers' or col == group_col:
                    continue

                values = group_df[col]

                # Treat explicitly listed categorical columns as non-numeric
                if col in categorical_columns or not pd.api.types.is_numeric_dtype(values):
                    value_counts = values.value_counts(dropna=False).to_dict()
                    stats = {
                        'value_counts': value_counts,
                        'n_nan': values.isna().sum()
                    }
                else:
                    stats = {
                        'mean': values.mean(),
                        'std': values.std(),
                        'count': values.count(),
                        'n_nan': values.isna().sum()
                    }

                result[f"{col}__{group_key}"] = stats

    return pd.DataFrame(result).T

def combine_redundant_columns(df, column_groups, new_column_names):
    """
    Combines multiple redundant columns into one by filling missing values in order,
    and drops the original redundant columns.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        column_groups (list of list of str): Each inner list contains redundant column names to combine.
        new_column_names (list of str): Names of the resulting combined columns (same length as column_groups).

    Returns:
        pd.DataFrame: Modified dataframe with new combined columns and original redundant columns dropped.
    """
    for group, new_col in zip(column_groups, new_column_names):
        combined = df[group[0]].copy()
        for col in group[1:]:
            combined = combined.fillna(df[col])
        df[new_col] = combined
        df.drop(columns=group, inplace=True)

    return df

def convert_maternal_education_num_to_string(score):
    if pd.isna(score):
        return "nan"
    elif score < 9:
        return "some_hs"
    elif score < 12:
        return "some_hs"
    elif score == 12:
        return "high_school"
    elif score <= 14:
        return "some_college"
    elif score == 16:
        return "college_degree"
    elif score <= 18:
        return "some_grad_level"
    elif score > 18:
        return "grad_degree"
    else:
        return "nan"

def add_IQ_ADOS(df_input, v24_v36_ADOS_filename, VSA_ADOS_filename, IQ_filename):
    df = df_input.copy()
    V24_V36_ADOS_df = pd.read_csv(v24_v36_ADOS_filename)
    VSA_ADOS_df = pd.read_csv(VSA_ADOS_filename)
    IQ_df = pd.read_csv(IQ_filename)

    V24_V36_ADOS_keep_columns=['Identifiers','V24 ADOS_Derived,severity_score_lookup', 'V36 ADOS_Derived,severity_score_lookup', 'V37Plus ADOS_Derived,severity_score_lookup']
    VSA_ADOS_keep_columns = ['Identifiers', 'VSA ados2_module1,severity_score_lookup','VSA ados2_module2,severity_score_lookup','VSA ados2_module3,severity_score_lookup']
    IQ_keep_columns=['Identifiers', 'VSA DAS_SA,Candidate_Age', 'V12 mullen,composite_standard_score','V24 mullen,composite_standard_score', 'VSA DAS_SA,GCA_STD_SCORE']

    V24_V36_ADOS_df = V24_V36_ADOS_df[V24_V36_ADOS_keep_columns]
    V24_V36_ADOS_df = V24_V36_ADOS_df.replace('.', np.nan)
    VSA_ADOS_df = VSA_ADOS_df[VSA_ADOS_keep_columns]
    VSA_ADOS_df = VSA_ADOS_df.replace('.', np.nan)
    IQ_df = IQ_df[IQ_keep_columns]
    IQ_df = IQ_df.replace('.', np.nan)

    # Create new severity score column for V24 V36 that collapses across multiple columns
    V24_V36_ADOS_df["V24_V36_ados_severity_score"] = (
        V24_V36_ADOS_df["V37Plus ADOS_Derived,severity_score_lookup"]
        .combine_first(V24_V36_ADOS_df["V36 ADOS_Derived,severity_score_lookup"])
        .combine_first(V24_V36_ADOS_df["V24 ADOS_Derived,severity_score_lookup"])
    )

    # Keep only Identifiers and the new score column
    V24_V36_ADOS_df = V24_V36_ADOS_df[["Identifiers", "V24_V36_ados_severity_score"]]

    # Create new severity score column for VSA that collapses across multiple columns
    VSA_ADOS_df["VSA_ados2_severity_score"] = (
        VSA_ADOS_df["VSA ados2_module3,severity_score_lookup"]
        .combine_first(VSA_ADOS_df["VSA ados2_module2,severity_score_lookup"])
        .combine_first(VSA_ADOS_df["VSA ados2_module1,severity_score_lookup"])
    )

    # Keep only Identifiers and the new score column
    VSA_ADOS_df = VSA_ADOS_df[["Identifiers", "VSA_ados2_severity_score"]]

    IQ_df = IQ_df[["Identifiers", "VSA DAS_SA,Candidate_Age", "V12 mullen,composite_standard_score", "V24 mullen,composite_standard_score",
                   "VSA DAS_SA,GCA_STD_SCORE"]]

    # Replace the literal string 'less than 25' with 25
    IQ_df["VSA DAS_SA,GCA_STD_SCORE"] = IQ_df["VSA DAS_SA,GCA_STD_SCORE"].replace("less than 25", 25).astype(float)

    # Left merge final df and ADOS df on Identifier
    final_df = (
        df.merge(V24_V36_ADOS_df, on="Identifiers", how="left")
        .merge(VSA_ADOS_df, on="Identifiers", how="left")
        .merge(IQ_df, on="Identifiers", how="left")
    )

    # Create new age for school age that takes DAS age and if it's missing, uss NIH Toolbox age
    final_df["Final_Age_School_Age"] = (
        final_df['VSA DAS_SA,Candidate_Age']
        .combine_first(final_df["Age_SchoolAge"])
            )
    final_df.drop(columns=["VSA DAS_SA,Candidate_Age", "Age_SchoolAge"], inplace=True)

    # Convert new columns to type float
    cols_to_float = ['V24_V36_ados_severity_score', 'VSA_ados2_severity_score', 'V12 mullen,composite_standard_score',
                     'V24 mullen,composite_standard_score','VSA DAS_SA,GCA_STD_SCORE', 'Final_Age_School_Age']

    final_df[cols_to_float] = final_df[cols_to_float].astype(float)

    return final_df

def add_race(final_df, race_filename, tsi_filename, vsa_filename):

    df1= pd.read_csv(race_filename)
    df2 = pd.read_csv(tsi_filename)
    df3 = pd.read_csv(vsa_filename)

    def filter_columns(df):
        keep = [col for col in df.columns if col == "Identifiers" or "race" in col.lower()]
        df_filtered = df[keep].copy()  # avoid SettingWithCopyWarning
        df_filtered = df_filtered.replace('.', np.nan, regex=False)
        return df_filtered

    # Example: apply to your 4 dataframes
    df1 = filter_columns(df1)
    df2 = filter_columns(df2)
    df3 = filter_columns(df3)

    # mapping dictionary for race codes
    race_map = {
        1: "white",
        2: "black_african_american",
        4: "asian",
        8: "american_indian_alaska_native",
        16: "native_hawaiian_pacific_islander",
        32: "other",
        0: "unknown_not_reported",
        64: "unknown_not_reported"
    }

    # Map nihtoolbox race codes to strings
    def map_race_codes(df, code_col, new_col="race_string"):
        df[new_col] = df[code_col].apply(
            lambda x: race_map.get(int(x), "more_than_one_race") if pd.notna(x) else "missing"
        )
        return df

    df3 = combine_vsa_columns(df3)
    df3 = map_race_codes(df3, "VSD-All NIHToolBox,Registration_Data_Race")
    df3.drop(columns=["VSD-All NIHToolBox,Registration_Data_Race"], inplace=True)

    # Merge df1, df2, df3 on "Identifiers" with full outer join
    merged = df1.merge(df2, on="Identifiers", how="outer") \
        .merge(df3, on="Identifiers", how="outer")

    # Replace all columns containing '@' to 'more_than_one_race'
    merged = merged.replace(r".*@.*", "more_than_one_race", regex=True)

    # List the columns in order (excluding Identifiers)
    cols_to_collapse = merged.columns[1:]  # all columns after Identifiers

    # Create new column with first non-NaN value from left to right
    merged["race"] = merged[cols_to_collapse].bfill(axis=1).iloc[:, 0]

    # Drop the old columns
    merged = merged.drop(columns=cols_to_collapse)

    merged["race"] = merged["race"].fillna("unknown_not_reported")

    # Merge merged dataframe with final_df, keeping only Identifiers from final_df
    merged_final = final_df.merge(merged, on="Identifiers", how="left")

    # Fill missing race_final values in final dataframe
    merged_final["race"] = merged_final["race"].fillna("unknown_not_reported")

    return merged_final

def add_missing_ages_from_brief2(final_data, brief2_ages):
    # Merge final_data with brief2_ages to get Calculated_Age
    merged = final_data.merge(
        brief2_ages[["Identifiers", "Calculated_Age"]],
        on="Identifiers",
        how="left"
    )

    # Replace Final_Age_School_Age with Calculated_Age where available
    merged["Final_Age_School_Age"] = merged["Calculated_Age"].combine_first(merged["Final_Age_School_Age"])

    # Drop the temporary Calculated_Age column
    merged = merged.drop(columns=["Calculated_Age"])

    return merged