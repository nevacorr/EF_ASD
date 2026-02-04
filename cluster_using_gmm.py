
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


def cluster_using_gmm(df, behavior_cols, group):

    df_group = df[df["Risk"] == group].copy()
    df_group = df_group.dropna(subset=behavior_cols)
    df_group.reset_index(inplace=True, drop=True)

    X = df_group[behavior_cols].values
    X_scaled = StandardScaler().fit_transform(X)

    bic = []
    models = {}

    for k in range(2, 6):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=42
        )
        gmm.fit(X_scaled)
        bic.append(gmm.bic(X_scaled))
        models[k] = gmm

    best_k = list(models.keys())[np.argmin(bic)]
    best_gmm = models[best_k]

    print(f"Selected number of clusters: {best_k}")

    df_group["cluster"] = best_gmm.predict(X_scaled)

    # Optional: soft assignments
    cluster_probs = best_gmm.predict_proba(X_scaled)

    cluster_profiles = (
        df_group
        .groupby("cluster")[behavior_cols]
        .mean()
    )

    print(cluster_profiles)

    cluster_counts = df_group['cluster'].value_counts().sort_index()
    print(cluster_counts)

    # --- Prepare LR group for plotting ---
    df_lr = df[df["Risk"] == "LR"].copy()
    df_lr = df_lr.dropna(subset=behavior_cols)
    df_lr['cluster'] = 'LR'

    # --- Combine HR clusters and LR for plotting ---
    df_plot = pd.concat([df_group, df_lr], ignore_index=True)

    # Reverse BRIEF2 scores so that higher = better
    # --- Reverse BRIEF2 scores so higher = better ---
    brief_cols = ["BRIEF2_GEC_T_score", "BRIEF2_shift_T_score",
                  "BRIEF2_inhibit_T_score", "BRIEF2_working_memory_T_score"]
    df_plot[brief_cols] = -df_plot[brief_cols] + df_plot[brief_cols].max(axis=0)

    # --- Scale all measures to z-scores across combined HR + LR ---
    scaler = StandardScaler()
    df_plot[behavior_cols] = scaler.fit_transform(df_plot[behavior_cols])

    # --- Melt for plotting ---
    df_long = df_plot.melt(
        id_vars='cluster',
        value_vars=behavior_cols,
        var_name='measure',
        value_name='zscore'
    )

    # --- Compute mean and SD per cluster/measure ---
    summary = df_long.groupby(['cluster', 'measure'])['zscore'].agg(
        mean='mean',
        sd='std'
    ).reset_index()

    # --- Ensure cluster labels are strings for consistent coloring ---
    summary['cluster'] = summary['cluster'].apply(lambda x: str(int(x)) if x != 'LR' else 'LR')

    # --- Define colors ---
    colors = {'0': 'blue', '1': 'red', 'LR': 'green'}

    # --- Plot with error bars ---
    plt.figure(figsize=(10, 6))

    for cluster in summary['cluster'].unique():
        sub = summary[summary['cluster'] == cluster]
        plt.errorbar(
            sub['measure'],
            sub['mean'],
            yerr=sub['sd'],
            label=f"Cluster {cluster}" if cluster != 'LR' else 'LR',
            marker='o',
            capsize=5,
            color=colors[cluster]
        )

    plt.xticks(rotation=45)
    plt.ylabel("Z-score (HR + LR)")
    plt.title(f"Behavioral Profiles by Cluster ({group} with LR reference)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Contingency table ---
    asds_by_cluster = pd.crosstab(df_group['cluster'], df_group['Group'])
    print("Counts per cluster:")
    print(asds_by_cluster)
    print("\n")

    # --- Percent HR+ (ASD) per cluster ---
    percent_HRp = pd.crosstab(df_group['cluster'], df_group['Group'], normalize='index') * 100
    print("Percent HR+ per cluster:")
    print(percent_HRp)
    print("\n")

    # --- Chi-square test ---
    chi2, p, dof, expected = chi2_contingency(asds_by_cluster)
    print(f"Chi-square test: chi2 = {chi2:.2f}, p = {p:.3f}")

    # Fisher exact test (only works for 2x2)
    if asds_by_cluster.shape == (2, 2):
        oddsratio, p_fisher = fisher_exact(asds_by_cluster)
        print(f"Fisher exact test: OR = {oddsratio:.2f}, p = {p_fisher:.3f}")

    # --- Bar plot of percent HR+ per cluster ---
    plt.figure(figsize=(6, 4))
    percent_HRp_plot = percent_HRp['HR+']  # column = HR+
    percent_HRp_plot.plot(kind='bar', color='skyblue')
    plt.ylabel("Percent HR+ (ASD)")
    plt.xlabel("Cluster")
    plt.title("ASD outcome by behavioral cluster")
    plt.ylim(0, 100)
    plt.xticks(rotation=0)
    plt.show()

    # --- Define groups ---
    clusters = ['0', '1', 'LR']  # must match the 'cluster' labels used in your plotting
    group_pairs = list(itertools.combinations(clusters, 2))  # all pairwise combinations

    # --- Prepare dictionary of data by cluster ---
    data_dict = {grp: df_plot[df_plot['cluster'] == grp] for grp in clusters}

    # --- Store results ---
    results = []

    for measure in behavior_cols:
        for grp1, grp2 in group_pairs:
            x = data_dict[grp1][measure]
            y = data_dict[grp2][measure]
            stat, p = mannwhitneyu(x, y, alternative='two-sided')
            results.append({'measure': measure,
                            'group1': grp1,
                            'group2': grp2,
                            'U': stat,
                            'p': p})

    # --- Convert to DataFrame ---
    results_df = pd.DataFrame(results)

    # --- Correct p-values for multiple comparisons (FDR) ---
    results_df['p_corrected'] = multipletests(results_df['p'], method='fdr_bh')[1]
    results_df['significant'] = results_df['p_corrected'] < 0.05

    # --- Show results ---
    print(results_df)

    return df_group, best_gmm, cluster_profiles