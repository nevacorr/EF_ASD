from dataclasses import asdict

import pandas as pd
import os
import numpy as np

from Utility_Functions_Demographics import simplify_dob_df, combine_vsa_columns, remove_invalid_anotb_data
from Utility_Functions_Demographics import remove_24mo_extra_ASD_DX_text, remove_extra_text_eftasks, remove_extra_ASD_DX_text
from Utility_Functions_Demographics import remove_extra_text_asd_diagnosis, combine_age_nihtoolbox, create_combined_dx_risk_column
from Utility_Functions_Demographics import replace_missing_with_nan, remove_fragx_downsyndrome_subj
from Utility_Functions_Demographics import remove_extra_ASD_DX_text, remove_extra_text_eftasks, convert_numeric_columns_to_numeric_type
from Utility_Functions_Demographics import remove_subj_no_behav_data, make_flanker_dccs_columns
from Utility_Functions_Demographics import make_and_plot_missing_data_map, write_missing_to_file
from Utility_Functions_Demographics import remove_Brief2_columns, calculate_nihtoolbox_age, combine_asd_dx
from Utility_Functions_Demographics import make_lists_of_columns_needed, combine_age_nihtoolbox
from plot_data_histograms import plot_data_histograms

working_dir = os.getcwd()

# Define location of data to import
datadir = '/Users/nevao/Documents/IBIS_EF/source data/Behav_Data/'

# Load executive function and demographic data
anotb = pd.read_csv(os.path.join(datadir, 'AnotB_clean.csv'))
brief1 = pd.read_csv(os.path.join(datadir, 'BRIEF1_UNC.csv'))
brief2 = pd.read_csv(os.path.join(datadir, 'BRIEF-2_7-1-24_data-2024-07-01T19_35_29.390Z.csv'))
dx = pd.read_csv(os.path.join(datadir, 'DSM_7-1-24_data-2024-07-01T21_05_03.559Z.csv'))
dx2 = pd.read_csv(os.path.join(datadir, 'New-11-22-21_data-2021-11-23T07_39_34.455Z.csv'))
nihtoolbox = pd.read_csv(os.path.join(datadir, 'NIH Toolbox_7-1-24_data-2024-07-01T19_40_36.204Z.csv'))
dob_risk_sex = pd.read_csv(os.path.join(datadir, 'DOB_sex_risk_11-8-24.csv'))
asd_diagnosis = pd.read_csv(os.path.join(datadir, 'IBIS 1 and 2_ASD Diagnosis-2024-11-22.csv'))
time1_demographics = pd.read_csv(os.path.join(datadir, 'IBIS 1 and 2 TSI demographic.csv'))
# Make lists of columns to keep or remove
(anotb_cols_to_remove, brief2_cols_to_keep, brief1_cols_to_keep, dx_cols_to_keep, dx2_cols_to_keep,
            nihtoolbox_cols_to_keep, asd_cols_to_keep, demot1_cols_to_keep) \
                    = make_lists_of_columns_needed(anotb, brief2, dx, dx2, nihtoolbox, asd_diagnosis, time1_demographics)

# Remove extra text from ASD_DX at 24months column
# dx2 = remove_24mo_extra_ASD_DX_text(dx2)

####### ---- DOB-Risk_Sex -----#######
# Combine columns from dob/risk/sex dataframe
dob_df_final = simplify_dob_df(dob_risk_sex)

# Remove or keep columns specified in lists created above
anotb = anotb.drop(columns=anotb_cols_to_remove)
brief1 = brief1.loc[:, brief1.columns.isin(brief1_cols_to_keep)]
# brief2 = brief2.loc[:, brief2.columns.isin(brief2_cols_to_keep)]
dx = dx.loc[:, dx.columns.isin(dx_cols_to_keep)]
dx2 = dx2.loc[:, dx2.columns.isin(dx2_cols_to_keep)]
nihtoolbox = nihtoolbox.loc[:, nihtoolbox.columns.isin(nihtoolbox_cols_to_keep)]
asd_diagnosis = asd_diagnosis.loc[:, asd_diagnosis.columns.isin(asd_cols_to_keep)]
asd_diagnosis.replace('No DSMIV ever administered', np.nan, inplace=True)
time1_demographics = time1_demographics.loc[:, time1_demographics.columns.isin(demot1_cols_to_keep)]

# Combine VSA and VSA-CD columns
# brief2 = combine_vsa_columns(brief2)
dx = combine_vsa_columns(dx)
nihtoolbox = combine_vsa_columns(nihtoolbox)
asd_diagnosis = combine_vsa_columns(asd_diagnosis)

# Remove extra text from asd_diagnosis dataframe and convert to a single column
asd_diagnosis = remove_extra_text_asd_diagnosis(asd_diagnosis)

# Change anotb invalid scores (indicated by validity score = 3) to NaN
# anotb = remove_invalid_anotb_data(anotb)

# Merge all dataframes
merged_demograph_behavior_df = (dob_df_final.merge(dx, on='Identifiers', how='outer')
                                .merge(time1_demographics, on='Identifiers', how='outer')
                                .merge(dx2, on='Identifiers',how='outer')
                                .merge(nihtoolbox, on='Identifiers',how='outer'))

# Replace missing values with NaN
merged_demograph_behavior_df = replace_missing_with_nan(merged_demograph_behavior_df)

# Remove all rows that have Down Syndrome Infant or Fragile X for VXX demographics,Project
IBIS_demograph_behavior_df = remove_fragx_downsyndrome_subj(merged_demograph_behavior_df)

# Save this dataframe
IBIS_demograph_behavior_df.to_csv('IBIS_merged_df_full_demographics.csv', index=None)

mystop=1

