import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu
import itertools
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def cluster_using_gmm(df, behavior_cols, group):
    # --------------------------
    # Subset HR group
    # --------------------------
    df_group = df[df["Risk"] == group].copy()
    df_group = df_group.dropna(subset=behavior_cols)
    df_group.reset_index(inplace=True, drop=True)

    # --------------------------
    # Standardize and compute PC1
    # --------------------------
    X_scaled = StandardScaler().fit_transform(df_group[behavior_cols])
    pca = PCA(n_components=1)
    df_group['PC1'] = pca.fit_transform(X_scaled).flatten()  # FIX: .flatten()

    # --------------------------
    # GMM on PC1 only (low/high groups)
    # --------------------------
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(df_group[['PC1']])
    df_group['cluster'] = gmm.predict(df_group[['PC1']]).astype(str)

    # Optional: reorder clusters so that 0 = low PC1, 1 = high PC1
    centroids = gmm.means_.flatten()
    if centroids[0] > centroids[1]:
        df_group['cluster'] = df_group['cluster'].replace({'0': '1', '1': '0'})

    # --------------------------
    # Prepare LR group
    # --------------------------
    df_lr = df[df["Risk"] == "LR"].copy()
    df_lr = df_lr.dropna(subset=behavior_cols)
    df_lr['cluster'] = 'LR'

    # --------------------------
    # Combine for plotting
    # --------------------------
    df_plot = pd.concat([df_group, df_lr], ignore_index=True)
    df_plot['cluster'] = df_plot['cluster'].astype(str)

    # --------------------------
    # Bar plot of cluster sizes
    # --------------------------
    cluster_counts = df_plot['cluster'].value_counts().reindex(['0', '1', 'LR'])
    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=cluster_counts.index,
        y=cluster_counts.values,
        palette={'0': '#1f77b4', '1': '#ff7f0e', 'LR': '#2ca02c'}
    )
    plt.xlabel("Cluster / Group")
    plt.ylabel("Number of subjects")
    plt.title(f"Number of subjects per cluster ({group} + LR)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.histplot(df_group, x='PC1', hue='cluster', palette={'0': 'blue', '1': 'red'}, bins=20, alpha=0.5,
                 stat='density')
    # plt.axvline(df_group[df_group['cluster'] == '0']['PC1'].max(), color='blue', linestyle='--',
    #             label='Cluster 0 threshold')
    # plt.axvline(df_group[df_group['cluster'] == '1']['PC1'].min(), color='red', linestyle='--',
    #             label='Cluster 1 threshold')
    plt.xlabel("PC1 (behavioral composite)")
    plt.ylabel("Density")
    plt.title("Distribution of PC1 by cluster")
    # plt.legend()
    plt.show()

    # --------------------------
    # Behavioral z-score plots
    # --------------------------
    # Reverse BRIEF2 so higher = better
    brief_cols = ["BRIEF2_GEC_T_score"]
    df_plot[brief_cols] = -df_plot[brief_cols] + df_plot[brief_cols].max(axis=0)

    # Z-score all behavioral measures
    scaler = StandardScaler()
    df_plot[behavior_cols] = scaler.fit_transform(df_plot[behavior_cols])

    # Melt for plotting
    df_long = df_plot.melt(id_vars='cluster', value_vars=behavior_cols,
                           var_name='measure', value_name='zscore')

    # Compute mean + 95% CI
    summary = df_long.groupby(['cluster', 'measure'])['zscore'].agg(
        mean='mean',
        ci95=lambda x: 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    ).reset_index()

    # Ensure cluster labels are strings
    summary['cluster'] = summary['cluster'].apply(lambda x: str(x) if x != 'LR' else 'LR')

    colors = {'0': 'blue', '1': 'red', 'LR': 'green'}

    plt.figure(figsize=(10, 6))
    for cluster in summary['cluster'].unique():
        sub = summary[summary['cluster'] == cluster]
        plt.errorbar(sub['measure'], sub['mean'], yerr=sub['ci95'],
                     marker='o', capsize=5, color=colors[cluster],
                     label=f"Cluster {cluster}" if cluster != 'LR' else 'LR')
    plt.xticks(rotation=45)
    plt.ylabel("Z-score (BRIEF2 sign reversed)")
    plt.title(f"Behavioral Profiles by Cluster (HR + LR reference)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------
    # Contingency table for ASD outcome
    # --------------------------
    asds_by_cluster = pd.crosstab(df_group['cluster'], df_group['Group'])
    print("Counts per cluster:")
    print(asds_by_cluster)
    print("\nPercent HR+ per cluster:")
    percent_HRp = asds_by_cluster.div(asds_by_cluster.sum(axis=1), axis=0) * 100
    print(percent_HRp)

    # -----------------------------
    # Bar plot: number of HR+ subjects per cluster
    # -----------------------------
    hr_counts = asds_by_cluster['HR+']  # counts of HR+ in each cluster

    plt.figure(figsize=(6, 4))
    sns.barplot(x=hr_counts.index, y=hr_counts.values, palette={'0': 'blue', '1': 'red'})
    plt.ylabel("Number of HR+ subjects")
    plt.xlabel("Cluster (Low vs High PC1)")
    plt.title("HR+ subjects per cluster")
    plt.ylim(0, hr_counts.max() + 5)
    plt.show()

    # Chi-square
    chi2, p, dof, expected = chi2_contingency(asds_by_cluster)
    print(f"Chi-square test: chi2={chi2:.2f}, p={p:.3f}")

    # Fisher exact (2x2 only)
    if asds_by_cluster.shape == (2, 2):
        oddsratio, p_fisher = fisher_exact(asds_by_cluster)
        print(f"Fisher exact test: OR={oddsratio:.2f}, p={p_fisher:.3f}")

    return df_group