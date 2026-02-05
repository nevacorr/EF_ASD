from scipy.stats import mannwhitneyu
import pandas as pd
from statsmodels.stats.multitest import multipletests

def evaluate_brain_struct_diff_between_clusters(df_clusters, df_hr_z, brain_cols, behavior_cols):

    brain_cols_z = [col + "_z" for col in brain_cols]

    df = (df_hr_z.merge(df_clusters, on="Identifiers", how="inner"))

    results = []

    for region in brain_cols_z:
        g1 = df.loc[df.cluster == 0, region].dropna()
        g2 = df.loc[df.cluster == 1, region].dropna()

        stat, p = mannwhitneyu(g1, g2, alternative='two-sided')

        results.append({
            'region': region,
            'p': p,
            'median_cluster1': g1.median(),
            'median_cluster2': g2.median()
        })

    res_df = pd.DataFrame(results)

    # FDR correction
    res_df['p_fdr'] = multipletests(res_df.p, method='fdr_bh')[1]

    mystop=1