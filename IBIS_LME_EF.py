
import pandas as pd
import os
import numpy as np
from plot_data_histograms import plot_data_histograms
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

working_dir = os.getcwd()

# Define location of data to import
datadir = '/home/toddr/neva/PycharmProjects/data_dir/IBIS_Behav_Brain'

# Load executive function and demographic data
anotb = pd.read_csv(os.path.join(datadir, 'AnotB_clean.csv'))
brief1 = pd.read_csv(os.path.join(datadir, 'BRIEF1_UNC.csv'))
brief2 = pd.read_csv(os.path.join(datadir, 'BRIEF-2_7-1-24_data-2024-07-01T19_35_29.390Z.csv'))
dx = pd.read_csv(os.path.join(datadir, 'DSM_7-1-24_data-2024-07-01T21_05_03.559Z.csv'))
dx2 = pd.read_csv(os.path.join(datadir, 'New-11-22-21_data-2021-11-23T07_39_34.455Z.csv'))
nihtoolbox = pd.read_csv(os.path.join(datadir, 'NIH Toolbox_7-1-24_data-2024-07-01T19_40_36.204Z.csv'))
dob_risk_sex = pd.read_csv(os.path.join(datadir, 'DOB_sex_risk_11-8-24.csv'))

####### ---- A not B-----#######
# Make a list of anotb columns to keep
anotb_cols_to_keep = ['Identifiers', 'V12prefrontal_taskCandidate_Age', 'V24prefrontal_taskCandidate_Age',
                      'AB_12_Percent', 'AB_24_Percent', 'V12_AB_validitycode', 'V24_AB_validitycode',
                      'AB_Reversals_12_Percent', 'AB_Reversals_24_Percent']
anotb_cols_to_remove = list(anotb.columns.difference(anotb_cols_to_keep))

####### ---- BRIEF2-----#######
# Make a list of brief 2 columns to keep
brief2_cols_to_keep = ['Identifiers', 'VSA BRIEF2_Parent,Candidate_Age', 'VSA-CVD BRIEF2_Parent,Candidate_Age']
brief2_cols_to_keep.extend([col for col in brief2.columns.tolist() if 'raw_score' in col or 'T_score' in col])
remove_list = ['VSA BRIEF2_Parent,T_score', 'VSA-CVD BRIEF2_Parent,T_score','VSA BRIEF2_Parent,raw_score',
               'VSA-CVD BRIEF2_Parent,raw_score']
brief2_cols_to_keep = [col for col in brief2_cols_to_keep if col not in remove_list]

####### ---- Brief1-----#######
# Make a list of brief1 columns to remove
brief1_cols_to_remove = ['TestDate', 'TestDescription', 'Grade', 'Relationship', 'TestFormTorP',
                         'Howwellknownteacheronly', 'Durationofrelationshipteacheronly', 'MissingItems']

####### ---- Dx -----#######
# Make list of dx columns to remove
dx_cols_to_remove = ['VSA demographics,ASD_DX', 'VSA-CVD demographics,ASD_DX']

####### ---- Dx2 -----#######
# Make list of dx2 columns to keep
dx2_substrings_to_keep = ['CandID', 'Cohort', 'Family_CandID1', 'Family_CandID2', 'Project', 'Relationship_type',
                          'Site', 'Status', 'ethnicity', 'race']
dx2_cols_to_keep = ['Identifiers']
####### Keep column if any of ths substrings is in each column name
dx2_cols_to_keep.extend([col for col in dx2.columns if any(sub in col for sub in dx2_substrings_to_keep)])

####### ---- NIH Toolbox -----#######
#Make a list of nih toolbox columns to keep
nihtoolbox_cols_to_keep = ['Identifiers']
nihtoolbox_substrings_to_keep = ['Date_taken', 'Inst_24', 'Inst_25', 'Age_Corrected_Standard_Score', 'Candidate_Age']
nihtoolbox_cols_to_keep.extend([col for col in nihtoolbox.columns if any(sub in col for sub in nihtoolbox_substrings_to_keep)])

####### ---- DOB-Risk_Sex -----#######
# Combine columns to create one dataframe with one Sex, one Risk and on DoB column for all study subjects
# Replace '.' with NaN for easier handling
dob_risk_sex.replace('.', np.nan, inplace=True)
# Find all columns that have DOB (there are ones for different ages)
dob_cols = [col for col in dob_risk_sex.columns if 'DoB' in col]
dob_df = dob_risk_sex[dob_cols].copy()
# Collapse DoB values from all DoB columns into one column. Do the same for Risk and Sex
dob_df_final = dob_risk_sex['Identifiers'].to_frame().copy()
dob_df_final['DoB'] = dob_df.apply(lambda row: row.dropna().iloc[0] if row.dropna().nunique() == 1 else pd.NA, axis=1)
risk_cols = [col for col in dob_risk_sex.columns if 'Risk' in col]
tmp_df = dob_risk_sex[risk_cols].copy()
dob_df_final['Risk'] = tmp_df.apply(lambda row: row.dropna().iloc[0] if row.dropna().nunique() == 1 else pd.NA, axis=1)
sex_cols = [col for col in dob_risk_sex.columns if 'Sex' in col]
tmp_df = dob_risk_sex[sex_cols].copy()
dob_df_final['Sex'] = tmp_df.apply(lambda row: row.dropna().iloc[0] if row.dropna().nunique() == 1 else pd.NA, axis=1)

# Remove or keep columns specified in lists created above
anotb = anotb.drop(columns=anotb_cols_to_remove)
brief1 = brief1.drop(columns=brief1_cols_to_remove)
brief2 = brief2.loc[:, brief2.columns.isin(brief2_cols_to_keep)]
dx = dx.drop(columns=dx_cols_to_remove)
dx2 = dx2.loc[:, dx2.columns.isin(dx2_cols_to_keep)]
nihtoolbox = nihtoolbox.loc[:, nihtoolbox.columns.isin(nihtoolbox_cols_to_keep)]

# Function to combine columns with 'VSA' and 'VSA-CVD'
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

# Combine VSA and VSA-CD columns
brief2 = combine_vsa_columns(brief2)
dx = combine_vsa_columns(dx)
nihtoolbox = combine_vsa_columns(nihtoolbox)

# For anotb, change all scores to NaN that have validitycode=3
# Columns are type object. Convert validity codes to integer type
anotb['V12_AB_validitycode'] = pd.to_numeric(anotb['V12_AB_validitycode'], errors='coerce')
anotb['V24_AB_validitycode'] = pd.to_numeric(anotb['V24_AB_validitycode'], errors='coerce')
anotb['V12_AB_validitycode'] = anotb['V12_AB_validitycode'].astype('Int64')
anotb['V24_AB_validitycode'] = anotb['V24_AB_validitycode'].astype('Int64')
# If validity code is 3, replace score with nan
anotb.loc[anotb['V12_AB_validitycode'] == 3, 'AB_12_Percent'] = np.nan
anotb.loc[anotb['V24_AB_validitycode'] == 3, 'AB_24_Percent'] = np.nan

# Remove validity code columns
anotb.drop(columns=['V12_AB_validitycode', 'V24_AB_validitycode'], inplace=True)

# Merge all dataframes
merged_demograph_behavior_df = (dob_df_final.merge(dx, on='Identifiers', how='outer').merge(dx2, on='Identifiers',
                        how='outer').merge(anotb, on='Identifiers', how='outer').merge(nihtoolbox, on='Identifiers',
                        how='outer').merge(brief2, on='Identifiers', how='outer'))

# Replace missing values with NaN
merged_demograph_behavior_df.replace('#NULL!', np.nan, inplace=True)
merged_demograph_behavior_df.replace('.', np.nan, inplace=True)
merged_demograph_behavior_df.replace('nan', np.nan, inplace=True)
merged_demograph_behavior_df.replace('NaN', np.nan, inplace=True)
merged_demograph_behavior_df.replace('Unknown', np.nan, inplace=True)

# Remove all rows that have Down Syndrome Infant or Fragile X for VXX demographics,Project
proj_columns = [col for col in merged_demograph_behavior_df.columns if 'Project' in col]
IBIS_demograph_behavior_df = merged_demograph_behavior_df[~merged_demograph_behavior_df[proj_columns]
    .applymap(lambda cell: 'Fragile' in cell or 'Down' in cell if pd.notna(cell) else False)
    .any(axis=1)].copy()

IBIS_demograph_behavior_df.drop(columns=proj_columns, inplace=True)

# Save this dataframe
IBIS_demograph_behavior_df.to_csv('IBIS_merged_df_full.csv', index=None)

# Remove columns that I won't use in the first analysis
dx_col_to_keep = ['Identifiers', 'VSD-All demographics,ASD_Ever_DSMIV']
dx_already_removed = ['VSD-All demographics,Project']
dx_cols_to_ignore = dx_col_to_keep + dx_already_removed
dx_cols_to_remove = list(dx.columns.difference(dx_cols_to_ignore))
dx2_cols_to_keep = ['Identifiers', 'V06 demographics,Risk', 'V06 demographics,Sex','V12 demographics,Risk',
                    'V12 demographics,Sex','V24 demographics,Risk', 'V24 demographics,Sex',
                    'V06 demographics,Project', 'V12 demographics,Project', 'V24 demographics,Project']
dx2_cols_to_remove = list(dx2.columns.difference(dx2_cols_to_keep))
all_cols_to_remove = dx_cols_to_remove + dx2_cols_to_remove
IBIS_demograph_behavior_df.drop(columns=all_cols_to_remove, inplace=True)

# Remove extra text in columns that have ASD diagnostic category
IBIS_demograph_behavior_df['test'] = np.where(IBIS_demograph_behavior_df['VSD-All demographics,ASD_Ever_DSMIV'].fillna('').str.contains('ASD+', regex=False), 'ASD+',
                                                                    np.where(IBIS_demograph_behavior_df['VSD-All demographics,ASD_Ever_DSMIV'].fillna('').str.contains('ASD-', regex=False), 'ASD-',
                                                                    np.nan))
test=IBIS_demograph_behavior_df.pop('test')
IBIS_demograph_behavior_df.insert(2, 'test',test)
IBIS_demograph_behavior_df.rename(columns={'test': 'ASD_Ever_DSMIV', 'VSD-All NIHToolBox,Inst_24': 'NIHToolBox,TestName1',
                                           'VSD-All NIHToolBox,Inst_25': 'NIHToolBox,TestName2',
                                           'VSD-All NIHToolBox,Date_taken': 'NIHToolBox,Date_taken'}, inplace=True)

# Remove extra text in columns that have name of executive function task at older ages
IBIS_demograph_behavior_df['NIHToolBox,TestName1'] = np.where(IBIS_demograph_behavior_df['NIHToolBox,TestName1'].fillna('').str.contains('Flanker', regex=False), 'Flanker',
                                                    np.where(IBIS_demograph_behavior_df['NIHToolBox,TestName1'].fillna('').str.contains('Dimensional', regex=False), 'DCCS',
                                                    np.nan))
IBIS_demograph_behavior_df['NIHToolBox,TestName2'] = np.where(IBIS_demograph_behavior_df['NIHToolBox,TestName2'].fillna('').str.contains('Flanker', regex=False), 'Flanker',
                                                    np.where(IBIS_demograph_behavior_df['NIHToolBox,TestName2'].fillna('').str.contains('Dimensional', regex=False), 'DCCS',
                                                    np.nan))

# Remove ever diagnosis column. We want to keep only last diagnosis
IBIS_demograph_behavior_df.drop(columns=['VSD-All demographics,ASD_Ever_DSMIV'], inplace=True)

# Get list of columns with data type 'object'. Convert the ones that should be numeric to numeric type
object_columns = IBIS_demograph_behavior_df.select_dtypes(include='object').columns.tolist()
categorical_columns = ['DoB','Identifiers', 'ASD_Ever_DSMIV', 'Risk', 'Sex', 'NIHToolBox,TestName1', 'NIHToolBox,TestName2',
                       'V06 demographics,Risk', 'V12 demographics,Risk', 'V24 demographics,Risk', 'V06 demographics,Sex',
                       'V12 demographics,Sex', 'V24 demographics,Sex', 'NIHToolBox,Date_taken', 'V06 demographics,Project',
                       'V12 demographics,Project', 'V24 demographics,Project', 'V12_AB_validitycode', 'V24_AB_validitycode']
cols_convert_to_numeric = [col for col in object_columns if col not in categorical_columns]
IBIS_demograph_behavior_df[cols_convert_to_numeric] = IBIS_demograph_behavior_df[cols_convert_to_numeric].apply(pd.to_numeric, errors='coerce')

# Remove rows that have all nans in non-demographic columns
cols_demographic = ['Identifiers', 'Risk', 'Sex', 'DoB', 'ASD_Ever_DSMIV', 'V12prefrontal_taskCandidate_Age',
                    'V24prefrontal_taskCandidate_Age']
cols_not_demographic = [col for col in IBIS_demograph_behavior_df.columns if col not in cols_demographic]
ctest = IBIS_demograph_behavior_df.loc[:, cols_not_demographic].copy()
ctest = ctest.replace(['nan', 'NaN'], np.nan)
IBIS_demograph_behavior_df.replace(['nan', 'NaN'], np.nan, inplace=True)
all_nan_rows = ctest[ctest.isna().all(axis=1)]
all_nan_rows_IBIS = IBIS_demograph_behavior_df[IBIS_demograph_behavior_df.loc[:, cols_not_demographic].isna().all(axis=1)]
ctest.dropna( how='all', inplace=True, ignore_index=True)
IBIS_demograph_behavior_df.dropna(subset=cols_not_demographic, how='all', inplace=True, ignore_index=True)

# Make columns for Flanker and DCCS score
IBIS_demograph_behavior_df['Flanker_Standard_Age_Corrected'] = IBIS_demograph_behavior_df.apply(lambda row:
                        row['VSD-All NIHToolBox,Score_Age_Corrected_Standard_Score_1'] if row['NIHToolBox,TestName1'] == 'Flanker'
                        else (row['VSD-All NIHToolBox,Score_Age_Corrected_Standard_Score_2'] if row['NIHToolBox,TestName1'] == 'DCCS'
                        else np.nan), axis=1)

IBIS_demograph_behavior_df['DCCS_Standard_Age_Corrected'] = IBIS_demograph_behavior_df.apply(lambda row:
                        row['VSD-All NIHToolBox,Score_Age_Corrected_Standard_Score_1'] if row['NIHToolBox,TestName1'] == 'DCCS'
                        else (row['VSD-All NIHToolBox,Score_Age_Corrected_Standard_Score_2'] if row['NIHToolBox,TestName1'] == 'Flanker'
                        else np.nan), axis=1)

IBIS_demograph_behavior_df.drop(columns=['VSD-All NIHToolBox,Score_Age_Corrected_Standard_Score_1',
                                         'VSD-All NIHToolBox,Score_Age_Corrected_Standard_Score_2',
                                         'NIHToolBox,TestName1', 'NIHToolBox,TestName2'], inplace=True)

# Plot histograms of all data
# plot_data_histograms(working_dir, IBIS_demograph_behavior_df.drop(columns=['Identifiers']))

# Make binary map indicating presence or absence of data
df_nan_nonnan = IBIS_demograph_behavior_df.applymap(lambda x: 1 if pd.notna(x) else 0).astype(int)
df_nan_nonnan.drop(columns=['Identifiers'], inplace=True)
cmap = LinearSegmentedColormap.from_list('bwcmap', ['white','blue'])

# Plot heatmap indicating which measures are present and which are missing for all subjects
fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(df_nan_nonnan, cmap=cmap, xticklabels=True, cbar=False, linewidths=0.5, ax=ax)
plt.xticks(rotation=45, ha='right')
plt.title('IBIS data set  white=missing  blue=available')
plt.gcf().subplots_adjust(bottom=0.8)
fig.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(f'{working_dir}/All_Behaviors_Missing_Data_Heatmap.png')
plt.show(block=False)

# Remove Brief2 columns
brief2_cols = [col for col in IBIS_demograph_behavior_df.columns if 'BRIEF2' in col]
IBIS_demograph_behavior_df.drop(columns=brief2_cols, inplace=True)

# Make heatmap again of missing or non-missing values, this time without BRIEF2 measures
df_nan_nonnan = IBIS_demograph_behavior_df.applymap(lambda x: 1 if pd.notna(x) else 0).astype(int)
df_nan_nonnan.drop(columns=['Identifiers'], inplace=True)

fig, ax = plt.subplots(figsize=(10, 20))
sns.heatmap(df_nan_nonnan, cmap=cmap, xticklabels=True, cbar=False, linewidths=0.5, ax=ax)
plt.xticks(rotation=45, ha='right')
plt.title('IBIS data set  white=missing  blue=available')
plt.gcf().subplots_adjust(bottom=0.8)
fig.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(f'{working_dir}/AnotB_NIHToolbox_Missing_Data_Heatmap.png')
plt.show(block=False)

IBIS_demograph_behavior_df.to_csv(f'{working_dir}/IBIS_behav_dataframe_demographics_AnotB_Flanker_DCCS.csv')

#Write to file all subject numbers with missing DoB
rows_no_dob = IBIS_demograph_behavior_df[IBIS_demograph_behavior_df['DoB'].isna()]
rows_no_asddx = IBIS_demograph_behavior_df[IBIS_demograph_behavior_df['ASD_Ever_DSMIV'].isna()]

rows_no_dob['Identifiers'].to_csv(f'{working_dir}/IBIS_subjects_with_no_DOB.csv', index=False)
rows_no_asddx['Identifiers'].to_csv(f'{working_dir}/IBIS_subjects_with_no_ASD.csv', index=False)

mystop=1

