from helper_functions_clustering import cluster_and_summarize
import pandas as pd
from load_data_for_ML import load_all_data

target = "BRIEF2_GEC_T_score"
metric = 'subcort_infant'

df = load_all_data()
brain_cols = ['thalamus', 'putamen', 'caudate', 'pallidum', 'hippocampus', 'amygdala', 'accumbens', 'ventralDC', 'left_caudate', 'right_putamen', 'left_hippocampus', 'right_amygdala']
ef_cols = ['inhibition', 'working_memory', 'flexibility', 'emotional_control', 'global_ef']

# 1) Δ (change) brain + EF from 24mo to school age
df_clusters, hr_summary, ef_means = cluster_and_summarize(
    df, brain_cols, ef_cols, cluster_type='combined', cluster_mode='delta',
    time1='24mo', time2='school_age', method='kmeans', max_clusters=5
)

# 2) Brain-only clustering (absolute values at school age)
df_clusters_brain, hr_summary_brain, _ = cluster_and_summarize(
    df, brain_cols, [], cluster_type='brain', cluster_mode='absolute',
    time2='school_age', method='kmeans', max_clusters=5
)

# 3) EF-only clustering (absolute values)
df_clusters_ef, hr_summary_ef, ef_means_ef = cluster_and_summarize(
    df, [], ef_cols, cluster_type='ef', cluster_mode='absolute',
    time2='school_age', method='kmeans', max_clusters=5
)
