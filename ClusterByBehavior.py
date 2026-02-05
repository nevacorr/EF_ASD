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

brain_metric = 'volume_infant'
#options 'volume_infant', 'volume_VSA', 'subcort_VSA', 'subcort_infant', 'ad_VSA', 'rd_VSA', 'md_VSA', 'fa_VSA'
#        'surface_area_VSA', 'cortical_thickness_VSA', 'subcort_infant+volume_infant'

working_dir = os.getcwd()

behavior_filename="/Users/nevao/R_Projects/IBIS_EF/processed_datafiles/ibis_subj_demographics_and_data_used_for_2025analysis_with_Brief2_subscales_with_brief1.csv"

behavior_df=pd.read_csv(behavior_filename)
behavior_df.drop(columns=["X"], inplace=True)
substr_drop = ['education', 'Candidate_Age', 'raw', 'mullen', 'SchoolAge']
cols_to_drop = [col for col in behavior_df if any(s in col for s in substr_drop)]
behavior_df.drop(columns=cols_to_drop, inplace=True)

demographics = pd.read_csv(os.path.join(working_dir, 'demographics_by_subject.csv'), usecols=lambda c: c != "Unnamed: 0")

# Columns to drop explicitly
cols_to_drop = ['Sex', 'Group']

# Columns to drop based on substrings (case-insensitive)
substr_drop = ['Score', 'IQ', 'race', 'education']
cols_substr = [col for col in demographics.columns if any(s.lower() in col.lower() for s in substr_drop)]

all_drop = cols_to_drop + cols_substr

demographics.drop(columns=all_drop, inplace=True)

demo_beh_df = demographics.merge(behavior_df, on='Identifiers', how='left')

mystop=1

behavior_cols = ["BRIEF2_GEC_T_score", "Flanker_Standard_Age_Corrected", "DCCS_Standard_Age_Corrected"]
# behavior_cols = ["BRIEF2_GEC_T_score", "BRIEF2_shift_T_score","BRIEF2_inhibit_T_score","BRIEF2_working_memory_T_score"]
# behavior_cols = ["BRIEF2_GEC_T_score", "BRIEF2_shift_T_score","BRIEF2_inhibit_T_score","BRIEF2_working_memory_T_score","Flanker_Standard_Age_Corrected", "DCCS_Standard_Age_Corrected"]
cov_cols = ['Sex, GroupFinal_Age_School_Age']

group= 'HR'
features = behavior_cols

df_clusters = cluster_using_gmm(demo_beh_df, behavior_cols, group)

brain_df = load_all_data()

substr_to_drop_brain = ['DX', 'Risk', 'AB_', 'BRIEF', 'Corrected']

cols_to_drop_brain = [col for col in brain_df.columns if any(s in col for s in substr_to_drop_brain)]

brain_df.drop(columns=cols_to_drop_brain, inplace=True)

brain_beh_df = demo_beh_df.merge(brain_df, on='Identifiers', how='left')

ml_df, brain_cols, cov_cols = create_input_for_ML(brain_df, brain_metric)

demographics = pd.read_csv(os.path.join(working_dir, 'demographics_by_subject.csv'), usecols=lambda c: c != "Unnamed: 0")
# Columns to drop explicitly
cols_to_drop = ['Sex', 'Group']

# Columns to drop based on substrings (case-insensitive)
substr_drop = ['Score', 'IQ', 'race', 'education']
cols_substr = [col for col in demographics.columns if any(s.lower() in col.lower() for s in substr_drop)]

# Combine
all_drop = cols_to_drop + cols_substr

demographics.drop(columns=all_drop, inplace=True)

cov_cols.remove('Site')
cov_cols.remove('Group')

final_brain_df = pd.merge(demographics, brain_df, on='Identifiers', how='outer')


# df = df[['CandID'] + [c for c in df.columns if c != 'CandID']]

ef_col = 'BRIEF2_GEC_T_score'

df_hr_z = calc_normative_data(final_brain_df, group_col='Group', lr_label='LR-', hr_labels=['HR+', 'HR-'],
                            brain_cols=brain_cols, ef_col=ef_col, covariates=cov_cols)

df_hr_z = df_hr_z.drop(columns=['CandID'])

results_ind_brain_regoins = evaluate_brain_struct_diff_between_clusters(df_clusters, df_hr_z, brain_cols, behavior_cols)

multi_variate_analsis(df_clusters, df_hr_z, brain_cols)

univariate_regression(df_clusters, df_hr_z, brain_cols, brain_metric)

mystop=1