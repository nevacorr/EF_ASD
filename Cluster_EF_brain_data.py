from helper_functions_clustering import cluster_and_summarize
import pandas as pd
from load_data_for_ML import load_all_data
from create_input_for_clustering import create_input_for_clustering
from helper_functions_clustering import compute_lr_correlation_matrix, plot_lr_heatmap

target = "BRIEF2_GEC_T_score"
metric = 'subcort_VSA'

df = load_all_data()

df, brain_cols, cov_cols = create_input_for_clustering(df, target, metric)

ef_cols = [target]

# corr_mat = compute_lr_correlation_matrix(X[brain_cols])
# plot_lr_heatmap(corr_mat,title="Left–Right Subcortical Volume Correlations")


# # 1) Δ (change) brain + EF from 24mo to school age
# df_clusters, hr_summary, ef_means = cluster_and_summarize(
#     df, brain_cols, ef_cols, cluster_type='combined', cluster_mode='delta',
#     time1='24mo', time2='school_age', method='kmeans', max_clusters=5
# )

# 2) Brain-only clustering (absolute values at school age)
df = df.dropna(subset=brain_cols, how='any')
df_clusters_brain, hr_summary_brain, _ = cluster_and_summarize(
    df, brain_cols, [], cluster_type='brain', cluster_mode='absolute',
    time2='school_age', method='kmeans', max_clusters=5
)

# 3) EF-only clustering (absolute values)
df_clusters_ef, hr_summary_ef, ef_means_ef = cluster_and_summarize(
    df, [], ef_cols, cluster_type='ef', cluster_mode='absolute',
    time2='school_age', method='kmeans', max_clusters=5
)
