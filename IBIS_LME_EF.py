from dataclasses import asdict

import pandas as pd
import os
import numpy as np

from Utility_Functions import simplify_dob_df, combine_vsa_columns, remove_invalid_anotb_data, \
    remove_24mo_extra_ASD_DX_text, remove_extra_text_eftasks, remove_extra_ASD_DX_text, remove_extra_text_asd_diagnosis
from Utility_Functions import replace_missing_with_nan, remove_fragx_downsyndrome_subj
from Utility_Functions import remove_extra_ASD_DX_text, remove_extra_text_eftasks, convert_numeric_columns_to_numeric_type
from Utility_Functions import remove_subj_no_behav_data, make_flanker_dccs_columns
from Utility_Functions import make_and_plot_missing_data_map, write_missing_to_file
from Utility_Functions import remove_Brief2_columns, calculate_nihtoolbox_age, combine_asd_dx
from Utility_Functions import make_lists_of_columns_needed
from plot_data_histograms import plot_data_histograms

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
asd_diagnosis = pd.read_csv(os.path.join(datadir, 'IBIS 1 and 2_ASD Diagnosis-2024-11-22(data-2024-11-22T19_37_29.csv'))

# Make lists of columns to keep or remove
(anotb_cols_to_remove, brief2_cols_to_keep, brief1_cols_to_remove, dx_cols_to_remove, dx2_cols_to_keep,
            nihtoolbox_cols_to_keep, asd_cols_to_keep) \
                    = make_lists_of_columns_needed(anotb, brief2, dx2, nihtoolbox, asd_diagnosis)

# Remove extra text from ASD_DX at 24months column
# dx2 = remove_24mo_extra_ASD_DX_text(dx2)

####### ---- DOB-Risk_Sex -----#######
# Combine columns from dob/risk/sex dataframe
dob_df_final = simplify_dob_df(dob_risk_sex)

# Remove or keep columns specified in lists created above
anotb = anotb.drop(columns=anotb_cols_to_remove)
brief1 = brief1.drop(columns=brief1_cols_to_remove)
brief2 = brief2.loc[:, brief2.columns.isin(brief2_cols_to_keep)]
dx = dx.drop(columns=dx_cols_to_remove)
dx2 = dx2.loc[:, dx2.columns.isin(dx2_cols_to_keep)]
nihtoolbox = nihtoolbox.loc[:, nihtoolbox.columns.isin(nihtoolbox_cols_to_keep)]
asd_diagnosis = asd_diagnosis.loc[:, asd_diagnosis.columns.isin(asd_cols_to_keep)]
asd_diagnosis.replace('No DSMIV ever administered', np.nan, inplace=True)

# Combine VSA and VSA-CD columns
brief2 = combine_vsa_columns(brief2)
dx = combine_vsa_columns(dx)
nihtoolbox = combine_vsa_columns(nihtoolbox)
asd_diagnosis = combine_vsa_columns(asd_diagnosis)

# Remove extra text from asd_diagnosis dataframe and convert to a single column
asd_diagnosis = remove_extra_text_asd_diagnosis(asd_diagnosis)

# Change anotb invalid scores (indicated by validity score = 3) to NaN
anotb = remove_invalid_anotb_data(anotb)

# Merge all dataframes
merged_demograph_behavior_df = (dob_df_final.merge(asd_diagnosis, on='Identifiers', how='outer')
                                .merge(dx, on='Identifiers', how='outer')
                                .merge(dx2, on='Identifiers',how='outer')
                                .merge(anotb, on='Identifiers', how='outer')
                                .merge(nihtoolbox, on='Identifiers',how='outer')
                                .merge(brief2, on='Identifiers', how='outer'))

# Replace missing values with NaN
merged_demograph_behavior_df = replace_missing_with_nan(merged_demograph_behavior_df)

# Remove all rows that have Down Syndrome Infant or Fragile X for VXX demographics,Project
IBIS_demograph_behavior_df = remove_fragx_downsyndrome_subj(merged_demograph_behavior_df)

# Save this dataframe
IBIS_demograph_behavior_df.to_csv('IBIS_merged_df_full.csv', index=None)

# Remove columns that I won't use in the first analysis
dx_col_to_keep = ['Identifiers', 'VSD-All demographics,ASD_Ever_DSMIV']
dx_already_removed = ['VSD-All demographics,Project']
dx_cols_to_ignore = dx_col_to_keep + dx_already_removed
dx_cols_to_remove = list(dx.columns.difference(dx_cols_to_ignore))
dx2_cols_to_keep = ['Identifiers', 'V06 demographics,Risk', 'V06 demographics,Sex','V12 demographics,Risk',
                    'V12 demographics,Sex','V24 demographics,Risk', 'V24 demographics,Sex',
                    'V06 demographics,Project', 'V12 demographics,Project', 'V24 demographics,Project',
                    'V24 demographics,ASD_DX']
dx2_cols_to_remove = list(dx2.columns.difference(dx2_cols_to_keep))
all_cols_to_remove = dx_cols_to_remove + dx2_cols_to_remove

IBIS_demograph_behavior_df = (
    IBIS_demograph_behavior_df
    .drop(columns=all_cols_to_remove)
    .pipe(remove_extra_ASD_DX_text)
    .pipe(remove_extra_text_eftasks)
    .drop(columns=['VSD-All demographics,ASD_Ever_DSMIV'])
    .pipe(combine_asd_dx)
    .pipe(convert_numeric_columns_to_numeric_type)
    .pipe(remove_subj_no_behav_data)
    .pipe(make_flanker_dccs_columns)
)

# Plot histograms of all data
# plot_data_histograms(working_dir, IBIS_demograph_behavior_df.drop(columns=['Identifiers']))

# Add column with NIH toolbox test age
IBIS_demograph_behavior_df = calculate_nihtoolbox_age(IBIS_demograph_behavior_df)

# Make binary map indicating presence or absence of data and plot as heatmap and save to file
make_and_plot_missing_data_map(IBIS_demograph_behavior_df, working_dir, 'All_Behaviors_Missing_Data_Heatmap',
                               figsize=(20, 20))
# Remove Brief2 columns
IBIS_demograph_behavior_df = remove_Brief2_columns(IBIS_demograph_behavior_df)

# Make binary map indicating presence or absence of data and plot as heatmap, without Brief2 and save
make_and_plot_missing_data_map(IBIS_demograph_behavior_df, working_dir, 'AnotB_NIHToolbox_Missing_Data_Heatmap',
                               figsize=(10,20))
# Write dataframe to file
IBIS_demograph_behavior_df.to_csv(f'{working_dir}/IBIS_behav_dataframe_demographics_AnotB_Flanker_DCCS.csv')

#Write to file all subject numbers with missing DoB or missing ASD Dx
write_missing_to_file(IBIS_demograph_behavior_df, working_dir)

mystop=1

