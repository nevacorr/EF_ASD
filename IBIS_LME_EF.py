
import pandas as pd
import os
import numpy as np

# Define location of data to import
datadir = '/home/toddr/neva/PycharmProjects/data_dir/IBIS_Behav_Brain'

# Load executive function and demographic data
anotb = pd.read_csv(os.path.join(datadir, 'AnotB_clean.csv'))
brief1 = pd.read_csv(os.path.join(datadir, 'BRIEF1_UNC.csv'))
brief2 = pd.read_csv(os.path.join(datadir, 'BRIEF-2_7-1-24_data-2024-07-01T19_35_29.390Z.csv'))
dx = pd.read_csv(os.path.join(datadir, 'DSM_7-1-24_data-2024-07-01T21_05_03.559Z.csv'))
dx2 = pd.read_csv(os.path.join(datadir, 'New-11-22-21_data-2021-11-23T07_39_34.455Z.csv'))
nihtoolbox = pd.read_csv(os.path.join(datadir, 'NIH Toolbox_7-1-24_data-2024-07-01T19_40_36.204Z.csv'))

# Define which columns to remove from spreadsheets
anotb_cols_to_remove = ['DCCID', 'V06demographicsProject', 'V06demographicsRisk', 'V06demographicsSite',
                        'V06demographicsStatus', 'V12demographicsProject']

brief2_cols_to_keep = ['Identifiers', 'VSA BRIEF2_Parent,Candidate_Age', 'VSA-CVD BRIEF2_Parent,Candidate_Age']
brief2_cols_to_keep.extend([col for col in brief2.columns.tolist() if 'raw_score' in col or 'T_score' in col])

brief1_cols_to_remove = ['TestDate', 'TestDescription', 'Grade', 'Relationship', 'TestFormTorP',
                         'Howwellknownteacheronly', 'Durationofrelationshipteacheronly', 'MissingItems']

dx_cols_to_remove = ['VSA demographics,ASD_DX', 'VSA-CVD demographics,ASD_DX', 'VSA demographics,Project',
                     'VSA-CVD demographics,Project']

dx2_substrings_to_keep = ['CandID', 'Cohort', 'Family_CandID1', 'Family_CandID2', 'Project', 'Relationship_type',
                          'Risk', 'Sex', 'Site', 'Status', 'ethnicity', 'race']
dx2_cols_to_keep = ['Identifiers']
# Return value of True if any of ths substrings is in each column name
dx2_cols_to_keep.extend([col for col in dx2.columns if any(sub in col for sub in dx2_substrings_to_keep)])

nihtoolbox_cols_to_keep = ['Identifiers']
nihtoolbox_substrings_to_keep = ['Inst_10', 'Age_Corrected_Standard_Score', 'RawScore',
                                 'Uncorrected_Standard_Score', 'Validity']
nihtoolbox_cols_to_keep.extend([col for col in nihtoolbox.columns if any(sub in col for sub in nihtoolbox_substrings_to_keep)])

# Remove columns not needed
anotb = anotb.drop(columns=anotb_cols_to_remove)
brief1 = brief1.drop(columns=brief1_cols_to_remove)
brief2 = brief2.loc[:, brief2.columns.isin(brief2_cols_to_keep)]
dx = dx.drop(columns=dx_cols_to_remove)
dx2 = dx2.loc[:, dx2.columns.isin(dx2_cols_to_keep)]
nihtoolbox = nihtoolbox.loc[:, nihtoolbox.columns.isin(nihtoolbox_cols_to_keep)]

# Replace nans indicated by '.' with NaN
anotb.replace('#NULL!', np.nan, inplace=True)
anotb.replace('.', np.nan, inplace=True)
brief2.replace('.', np.nan, inplace=True)
dx.replace('.', np.nan, inplace=True)
nihtoolbox.replace('.', np.nan, inplace=True)

# Function to combine columns with 'VSA' and 'VSA-CVD'
def combine_vsa_columns(df):
    # Iterate through all column names
    for col in df.columns:
        if 'VSA-CVD' in col:
            vsa_col = col.replace('VSA-CVD', 'VSA')
            if vsa_col in df.columns:
                new_col = col.replace('VSA-CVD', 'VSD-All')
                df[new_col] = df[vsa_col].fillna(df[col])
                df = df.drop(columns=[vsa_col, col])
    return df

# Combine VSA and VSA-CD columns
# brief2_orig = brief2
brief2 = combine_vsa_columns(brief2)
dx = combine_vsa_columns(dx)
nihtoolbox = combine_vsa_columns(nihtoolbox)

merged_demograph_behavior_df = (dx.merge(dx2, on='Identifiers', how='outer').merge(anotb, on='Identifiers', how='outer').merge(brief2, on='Identifiers',
                                                        how='outer').merge(nihtoolbox, on='Identifiers', how='outer'))

IBIS_demograph_behavior_df = merged_demograph_behavior_df.dropna(axis=1, how='all')

# Save this as dataframe that has variables that I may ever want to look at
IBIS_demograph_behavior_df.to_csv('IBIS_merged_df.csv', index=None)

# Remove columns that I won't use in the first analysis
dx_col_to_keep = ['Identifiers', 'VSD-All demographics,ASD_Ever_DSMIV']
dx_cols_to_remove = list(dx.columns.difference(dx_col_to_keep))

dx2_cols_to_keep = ['Identifiers', 'V24 demographics,Risk', 'V24 demographics,Sex']
dx2_cols_to_remove = list(dx2.columns.difference(dx2_cols_to_keep))

anotb_cols_to_keep = ['Identifiers', 'AB_12_Percent', 'AB_24_Percent', 'V12_AB_validitycode', 'V24_AB_validitycode', 'AB_Reversals_12_Percent', 'AB_Reversals_24_Percent']
anotb_cols_to_remove = list(anotb.columns.difference(anotb_cols_to_keep))
anotb_cols_in_IBISdf = [cols for cols in anotb_cols_to_remove if cols in IBIS_demograph_behavior_df]

all_cols_to_remove = dx_cols_to_remove + dx2_cols_to_remove + anotb_cols_in_IBISdf

IBIS_demograph_behavior_df.drop(columns=all_cols_to_remove, inplace=True)

IBIS_demograph_behavior_df['test'] = np.where(IBIS_demograph_behavior_df['VSD-All demographics,ASD_Ever_DSMIV'].fillna('').str.contains('ASD+', regex=False), 'ASD+',
                                                                    np.where(IBIS_demograph_behavior_df['VSD-All demographics,ASD_Ever_DSMIV'].fillna('').str.contains('ASD-', regex=False), 'ASD-',
                                                                    np.nan))
test=IBIS_demograph_behavior_df.pop('test')
IBIS_demograph_behavior_df.insert(2, 'test',test)
IBIS_demograph_behavior_df.rename(columns={'test': 'ASD_Ever_DSMIV', 'V24 demographics,Risk': 'Risk',
                                           'V24 demographics,Sex': 'Sex', 'VSD-All NIHToolBox,Inst_10': 'NIHToolBox,TestName'}, inplace=True)
IBIS_demograph_behavior_df['NIHToolBox,TestName'] = np.where(IBIS_demograph_behavior_df['NIHToolBox,TestName'].fillna('').str.contains('Flanker', regex=False), 'Flanker',
                                                    np.where(IBIS_demograph_behavior_df['NIHToolBox,TestName'].fillna('').str.contains('Dimensional', regex=False), 'DCCS',
                                                    np.nan))
IBIS_demograph_behavior_df.drop(columns=['VSD-All demographics,ASD_Ever_DSMIV'], inplace=True)
mystop=1

