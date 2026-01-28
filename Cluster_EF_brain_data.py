import os

from matplotlib.cbook import delete_masked_points

from helper_functions_clustering import cluster_and_summarize
import pandas as pd
from load_data_for_ML import load_all_data
from create_input_for_clustering import create_input_for_clustering
from cluster_hr_group import cluster_hr_group
from helper_functions_clustering import compute_lr_correlation_matrix, plot_lr_heatmap
from cluster_deviations_from_norm import normative_clustering
import os

target = "BRIEF2_GEC_T_score"
metric = 'subcort_VSA'
working_dir = os.getcwd()

demographics = pd.read_csv(os.path.join(working_dir, 'demographics_by_subject.csv'), usecols=lambda c: c != "Unnamed: 0")
# Columns to drop explicitly
cols_to_drop = ['Sex', 'Group']

# Columns to drop based on substrings (case-insensitive)
substr_drop = ['Score', 'IQ', 'race', 'education']
cols_substr = [col for col in demographics.columns if any(s.lower() in col.lower() for s in substr_drop)]

# Combine
all_drop = cols_to_drop + cols_substr

demographics.drop(columns=all_drop, inplace=True)

df = load_all_data()

df, brain_cols, cov_cols = create_input_for_clustering(df, target, metric)

cov_cols.remove('Site')
cov_cols.remove('Group')

df = pd.merge(demographics, df, on='Identifiers', how='outer')
df.drop(columns=['Identifiers'], inplace=True)

df = df[['CandID'] + [c for c in df.columns if c != 'CandID']]

ef_cols = target

Age = "Final_Age_School_Age" if "BRIEF2" in target else None

cov_cols.append('Final_Age_School_Age')

# corr_mat = compute_lr_correlation_matrix(X[brain_cols])
# plot_lr_heatmap(corr_mat,title="Left–Right Subcortical Volume Correlations")


# # 1) Δ (change) brain + EF from 24mo to school age
# df_clusters, hr_summary, ef_means = cluster_and_summarize(
#     df, brain_cols, ef_cols, cluster_type='combined', cluster_mode='delta',
#     time1='24mo', time2='school_age', method='kmeans', max_clusters=5
# )

# 2) Brain-only clustering (absolute values at school age)
# df = df.dropna(subset=brain_cols, how='any')
# df_clusters_brain, hr_summary_brain, _ = cluster_and_summarize(
#     df, brain_cols, [], cluster_type='brain', cluster_mode='absolute',
#     time2='school_age', method='kmeans', max_clusters=5
# )
#
# # 3) EF-only clustering (absolute values)
# df_clusters_ef, hr_summary_ef, ef_means_ef = cluster_and_summarize(
#     df, [], ef_cols, cluster_type='ef', cluster_mode='absolute',
#     time2='school_age', method='kmeans', max_clusters=5
# )


# df_hr_clusters, cluster_means, ef_means = cluster_hr_group(
#     df,
#     dx_group='HR',
#     brain_cols=brain_cols,
#     ef_cols=ef_cols,
#     max_clusters=4
# )

df_hr_clustered, cluster_means, ef_means = normative_clustering(df,
                                                                group_col='Group',
                                                                lr_label='LR-',
                                                                hr_labels=['HR+', 'HR-'],
                                                                brain_cols=brain_cols,
                                                                ef_cols=ef_cols,
                                                                covariates=cov_cols,
                                                                n_clusters=2)