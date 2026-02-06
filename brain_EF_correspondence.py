from scipy.stats import mannwhitneyu
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
import pandas as pd
from statsmodels.multivariate.manova import MANOVA
import scipy.stats as stats
import statsmodels.stats.multitest as smm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.stats import pearsonr


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

    # ==========================
    # CLUSTER-BASED ANALYSIS
    # ==========================

    # Fit MANOVA: all brain regions as dependent variables, cluster as independent
    maov = MANOVA.from_formula(' + '.join(brain_cols_z) + ' ~ cluster', data=df)
    print(maov.mv_test())

    # -------------------------
    # LDA (CVA) for 2 clusters
    # -------------------------
    X_brain = df[brain_cols_z].values
    y_cluster = df['cluster'].values

    lda = LDA(n_components=1)  # 1 canonical axis for 2 clusters
    X_lda = lda.fit_transform(X_brain, y_cluster)

    df['Canonical1'] = X_lda[:, 0]

    # -------------------------
    # Plot clusters along canonical axis
    # -------------------------
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='cluster', y='Canonical1', data=df, palette={'0':'#1f77b4','1':'#ff7f0e'})
    sns.stripplot(x='cluster', y='Canonical1', data=df, color='k', alpha=0.6, jitter=True)
    plt.title("Clusters separated along canonical brain axis")
    plt.ylabel("Canonical axis (brain volumes)")
    plt.xlabel("EF cluster")
    plt.tight_layout()
    plt.show(block=False)

    # -------------------------
    # Show top regions driving separation
    # -------------------------
    coef = pd.Series(lda.coef_[0], index=brain_cols_z).sort_values(key=abs, ascending=False)
    top_regions = coef.head(10)

    plt.figure(figsize=(8, 4))
    sns.barplot(x=top_regions.values, y=top_regions.index, palette="viridis")
    plt.title("Top brain regions driving cluster separation (CVA)")
    plt.xlabel("Canonical coefficient")
    plt.tight_layout()
    plt.show(block=False)

    print("\nTop contributing brain regions:\n", top_regions)

    """
    Plots top regions driving canonical separation, colored by direction.
    lda: fitted LinearDiscriminantAnalysis object
    brain_cols_z: list of brain column names (z-scored)
    top_n: number of top contributing regions to show
    """

    top_n=10
    pc1_col="PC1"

    # Get canonical coefficients as Series
    coef = pd.Series(lda.coef_[0], index=brain_cols_z)

    # Select top N by absolute value
    top_idx = coef.abs().sort_values(ascending=False).head(top_n).index
    coef_top = coef.loc[top_idx]

    # Assign colors: blue = favors higher EF, red = favors lower EF
    colors = ['blue' if x > 0 else 'red' for x in coef_top.values]

    # Plot
    plt.figure(figsize=(8, 5))
    sns.barplot(x=coef_top.values, y=coef_top.index, palette=colors)
    plt.axvline(0, color='k', linestyle='--')
    plt.xlabel("Canonical coefficient")
    plt.ylabel("Brain region")
    plt.title(f"Top {top_n} brain regions driving cluster separation\nBlue = higher EF, Red = lower EF")
    plt.tight_layout()
    plt.show(block=False)

    print("\nCanonical coefficients of top regions:\n", coef_top)

    # ============================================================
    # PC1-BASED DIMENSIONAL ANALYSIS (ADDED)
    # ============================================================

    print("\n===== PC1-based dimensional brain–behavior analysis =====")

    # -------------------------
    # Multivariate regression: brain ~ PC1
    # (MANOVA analogue for continuous EF)
    # -------------------------
    maov_pc1 = MANOVA.from_formula(
        ' + '.join(brain_cols_z) + f' ~ {pc1_col}', data=df
    )
    print(maov_pc1.mv_test())

    # -------------------------
    # Canonical brain axis associated with PC1
    # (project brain data onto direction maximally correlated with PC1)
    # -------------------------

    # Center variables
    X = df[brain_cols_z].values
    X -= X.mean(axis=0)

    pc1 = df[pc1_col].values.reshape(-1, 1)
    pc1 -= pc1.mean()

    # Compute brain loadings proportional to covariance with PC1
    brain_loadings = (X.T @ pc1).flatten()

    # Normalize for interpretability
    brain_loadings /= np.linalg.norm(brain_loadings)

    df['Brain_PC1_axis'] = X @ brain_loadings

    # -------------------------
    # Plot PC1 vs brain canonical axis
    # -------------------------
    plt.figure(figsize=(6, 6))
    sns.regplot(x=pc1.flatten(), y=df['Brain_PC1_axis'],
                scatter_kws=dict(alpha=0.7), ci=None)

    r_val, p_val = pearsonr(df['PC1'], df['Brain_PC1_axis'])
    plt.xlabel("Behavioral PC1 (Executive Function)")
    plt.ylabel("Brain canonical axis")
    plt.title(f"Dimensional EF–Brain Association\n(r = {r_val:.2f}, p = {p_val:.3f})")
    plt.tight_layout()
    plt.show(block=False)

    # -------------------------
    # Top regions associated with PC1
    # -------------------------
    coef_pc1 = pd.Series(brain_loadings, index=brain_cols_z)
    top_pc1 = coef_pc1.abs().sort_values(ascending=False).head(10).index
    coef_pc1_top = coef_pc1.loc[top_pc1]

    colors = ['blue' if x > 0 else 'red' for x in coef_pc1_top.values]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=coef_pc1_top.values, y=coef_pc1_top.index, palette=colors)
    plt.axvline(0, color='k', linestyle='--')
    plt.xlabel("Brain–PC1 loading")
    plt.ylabel("Brain region")
    plt.title("Top brain regions associated with EF (PC1)\nBlue = higher EF")
    plt.tight_layout()
    plt.show(block=False)

    print("\nTop PC1-associated brain regions:\n", coef_pc1_top)

    mystop=1