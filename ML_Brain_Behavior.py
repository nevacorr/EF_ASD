from create_input_for_ML import create_input_for_ML
from normative_model_brain_data import calc_normative_data
import os
import  numpy as np
from load_data_for_ML import load_all_data
import pandas as pd
from cluster_using_gmm import cluster_using_gmm
from load_data_for_ML import load_all_data
from brain_EF_correspondence import evaluate_brain_struct_diff_between_clusters, univariate_regression
from brain_EF_correspondence import multi_variate_analsis
from perform_pls_regression import perform_pls_regression

perform_norm_modeling = False
brain_metric = 'subcort_VSA'
ef_col = 'BRIEF2_GEC_T_score'
# ef_col = 'Flanker_Standard_Age_Corrected'
#options 'volume_infant', 'volume_VSA', 'subcort_VSA', 'subcort_infant', 'ad_VSA', 'rd_VSA', 'md_VSA', 'fa_VSA'
#        'surface_area_VSA', 'cortical_thickness_VSA', 'subcort_infant+volume_infant'

working_dir = os.getcwd()

# Load EF data. Remove columns with age, parent education, non-standardized scores, IQ (mullen)
# Keep group, sex, final dx, and risk in this dataframe.
# Note that age is removed because this dataframe does not have complete ages for all school age subjects
behavior_filename="/Users/nevao/R_Projects/IBIS_EF/processed_datafiles/ibis_subj_demographics_and_data_used_for_2025analysis_with_Brief2_subscales_with_brief1.csv"
behavior_df=pd.read_csv(behavior_filename)
behavior_df.drop(columns=["X"], inplace=True)
substr_drop = ['education', 'Candidate_Age', 'raw', 'mullen', 'SchoolAge']
cols_to_drop = [col for col in behavior_df if any(s in col for s in substr_drop)]
behavior_df.drop(columns=cols_to_drop, inplace=True)

##### Load demographics file to get age. Remove all other columns except Identifier  #####
demographics = pd.read_csv(os.path.join(working_dir, 'demographics_by_subject.csv'), usecols=lambda c: c != "Unnamed: 0")
# Columns to drop explicitly
cols_to_drop = ['Sex', 'Group']
# Columns to drop based on substrings (case-insensitive)
substr_drop = ['Score', 'IQ', 'race', 'education']
cols_substr = [col for col in demographics.columns if any(s.lower() in col.lower() for s in substr_drop)]
all_drop = cols_to_drop + cols_substr
demographics.drop(columns=all_drop, inplace=True)

##### Merge age dataframe with ef data frame  #####
demo_beh_df = demographics.merge(behavior_df, on='Identifiers', how='left')

##### Load file with all brain and ef data. Remove ef data, and demographics #####
brain_df = load_all_data()
substr_to_drop_brain = ['DX', 'Risk', 'AB_', 'BRIEF', 'Corrected', 'Group', 'Sex', 'Group_HR+', 'Group_HR-', 'Group_LR-']
cols_to_drop_brain = [col for col in brain_df.columns if any(s in col for s in substr_to_drop_brain)]
brain_df.drop(columns=cols_to_drop_brain, inplace=True)

# Merge brain data with ef and demographic data
brain_beh_df = demo_beh_df.merge(brain_df, on='Identifiers', how='left')

# Based on chosen bran metric at top of script, return brain column names and non-brain covariate column names.
# Include ICV as covariate if brain metric is volume or surface area
# Also add site as a column to df
final_brain_df, brain_cols, cov_cols = create_input_for_ML(brain_beh_df, brain_metric)

if brain_metric == 'volume_VSA':
    brain_cols = [col for col in brain_cols if "Frontal" in col]

# Recode Sex as numeric
final_brain_df['Sex'] = final_brain_df['Sex'].replace({'Female': 0, 'Male': 1}).astype('Int64')

if perform_norm_modeling:
    # Calculate z_scores for HR subjects
    df_hr_z = calc_normative_data(final_brain_df, group_col='Group', lr_label='LR-', hr_labels=['HR+', 'HR-'],
                            brain_cols=brain_cols, covariates=['Sex', 'Final_Age_School_Age', 'ICV_vol_VSA'])

    df_hr_z = df_hr_z.drop(columns=['CandID'])

else:
    df_hr_z = final_brain_df[["Identifiers"] + brain_cols]

perform_pls_regression(final_brain_df, brain_cols, df_hr_z, ef_col, perform_norm_modeling)
