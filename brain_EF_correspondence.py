from scipy.stats import mannwhitneyu
import pandas as pd
from statsmodels.stats.multitest import multipletests
import pandas as pd
from statsmodels.multivariate.manova import MANOVA
import scipy.stats as stats
import statsmodels.stats.multitest as smm
import matplotlib.pyplot as plt
import seaborn as sns

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

    return res_df

def univariate_regression(df_clusters, df_hr_z, brain_cols, brain_metric):

    if brain_metric == 'volume_VSA':
        brain_cols = [col for col in brain_cols if "Frontal" in col]

    brain_cols_z = [col + "_z" for col in brain_cols]

    df = (df_hr_z.merge(df_clusters, on="Identifiers", how="inner"))

    results = []
    for col in brain_cols_z:
        r, p = stats.pearsonr(df[col], df['PC1'])
        results.append({'region': col, 'r': r, 'p_uncorrected': p})

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # FDR correction
    rej, p_fdr = smm.fdrcorrection(results_df['p_uncorrected'])
    results_df['p_fdr'] = p_fdr
    results_df['significant'] = rej

    # Sort by absolute correlation
    results_df = results_df.sort_values('r', key=abs, ascending=False)

    # --- Scatter plots for significant regions only ---
    sig_df = results_df[results_df['significant']]
    sig_df = results_df[results_df['p_uncorrected'] <  0.05]

    for _, row in sig_df.iterrows():
        col = row['region']
        plt.figure(figsize=(6,4))
        sns.scatterplot(
            data=df,
            x=col,
            y='PC1',
            hue='cluster',   # optional: color by cluster
            palette={0:'#1f77b4', 1:'#ff7f0e'},
            s=60,
            alpha=0.8
        )
        # Optional: add regression line
        sns.regplot(
            data=df,
            x=col,
            y='PC1',
            scatter=False,
            color='black',
            line_kws={'linewidth':1.5},
            ci=None
        )
        plt.title(f'PC1 vs {col} (p = {row["p_uncorrected"]:.3f} corr p = {row["p_fdr"]:.3f})')
        plt.xlabel(f'{col} (z-score)')
        plt.ylabel('PC1 (EF)')
        plt.tight_layout()
        plt.show()

    print(results_df)

def multi_variate_analsis(df_clusters, df_hr_z, brain_cols):

    brain_cols_z = [col + "_z" for col in brain_cols]

    df = (df_hr_z.merge(df_clusters, on="Identifiers", how="inner"))

    # Fit MANOVA: all brain regions as dependent variables, cluster as independent
    maov = MANOVA.from_formula(' + '.join(brain_cols_z) + ' ~ cluster', data=df)
    print(maov.mv_test())

    mystop=1