
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

    # --- Cluster profiles ---
    cluster_profiles = (
        df_group
        .groupby("cluster")[behavior_cols]
        .mean()
    )
    print(cluster_profiles)

    # --- Cluster sizes ---
    cluster_counts = df_group['cluster'].value_counts().sort_index()
    print(cluster_counts)

    # --- Silhouette score ---
    sil_score = silhouette_score(X_scaled, df_group['cluster'])
    print(f"Silhouette score for {best_k} clusters: {sil_score:.3f}")

    # --- PCA visualization ---
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    explained_var = pca.explained_variance_ratio_[0]
    print(f"PC1 explains {explained_var * 100:.1f}% of variance")

    df_group['PC1'] = X_pca[:,0]
    df_group['PC2'] = X_pca[:,1]

    # --- Plot PC1 by cluster ---
    plt.figure(figsize=(8, 6))
    sns.stripplot(
        data=df_group,
        x='cluster',
        y='PC1',
        palette={'0': '#1f77b4', '1': '#ff7f0e'},
        size=8,
    )
    plt.title(f"PC1 of EF tasks by cluster ({group} group only)\nExplained variance: {explained_var * 100:.1f}%")
    plt.xlabel("Cluster")
    plt.ylabel("PC1 (EF)")
    plt.tight_layout()
    plt.show(block=False)

    plt.figure(figsize=(8,6))
    sns.scatterplot(
        data=df_group,
        x='PC1', y='PC2',
        hue='cluster',
        palette = {0:'#1f77b4', 1:'#ff7f0e'},
        s=80,
        alpha=0.8
    )
    plt.title(f"PCA of EF data ({group} group only)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.show(block=False)

    # PC1 loadings
    pc1_loadings = pd.Series(
        pca.components_[0],
        index=behavior_cols
    ).sort_values(key=np.abs, ascending=False)

    print(pc1_loadings)

    # PC2 loadings
    pc2_loadings = pd.Series(
        pca.components_[1],
        index=behavior_cols
    ).sort_values(key=np.abs, ascending=False)

    print(pc2_loadings)

    # --- Prepare LR group for plotting ---
    df_lr = df[df["Risk"] == "LR"].copy()
    df_lr = df_lr.dropna(subset=behavior_cols)
    df_lr['cluster'] = 'LR'

    # --- Make a copy of the cluster column for plotting ---
    df_counts = pd.concat([df_group, df_lr], ignore_index=True)
    df_counts['cluster_plot'] = df_counts['cluster'].astype(str)

    # Count subjects per cluster/group
    cluster_counts = df_counts['cluster_plot'].value_counts()

    # Reorder for plotting (strings)
    cluster_order = ['0', '1', 'LR']
    cluster_counts = cluster_counts.reindex(cluster_order)

    # --- Bar plot ---
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
    plt.show(block=False)

    # ================================
    # Z SCORES PLOT (BRIEF2 + Flanker + DCCS)
    # ================================

    # --- Combine HR clusters and LR for plotting ---
    df_plot = pd.concat([df_group, df_lr], ignore_index=True)
    df_plot['cluster'] = df_plot['cluster'].astype(str)

    # Reverse BRIEF2 scores so that higher = better
    # --- Reverse BRIEF2 scores so higher = better ---
    brief_cols = ["BRIEF2_GEC_T_score"]
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
        ci95=lambda x: 1.96 * x.std(ddof=1) / np.sqrt(len(x))
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
            yerr=sub['ci95'],
            label=f"Cluster {cluster}" if cluster != 'LR' else 'LR',
            marker='o',
            capsize=5,
            color=colors[cluster]
        )

    plt.xticks(rotation=45)
    plt.ylabel("Z-score (BRIEF2 sign reversed)")
    plt.title(f"Behavioral Profiles (Z-score, BRIEF2 sign reversed) by Cluster (HR with LR reference)")
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)

    # ================================
    # RAW SCORES PLOT (BRIEF2 + Flanker + DCCS)
    # ================================

    df_plot = pd.concat([df_group, df_lr], ignore_index=True)
    df_plot['cluster'] = df_plot['cluster'].astype(str)

    # Reverse BRIEF2 scores so that higher = better
    # --- Reverse BRIEF2 scores so higher = better ---
    df_plot[brief_cols] = -df_plot[brief_cols] + df_plot[brief_cols].max(axis=0)

    # Melt for plotting
    df_raw_long = df_plot.melt(
        id_vars="cluster",
        value_vars=behavior_cols,
        var_name="measure",
        value_name="raw_score"
    )

    # Compute mean + 95% CI
    raw_summary = (
        df_raw_long
        .groupby(["cluster", "measure"])["raw_score"]
        .agg(
            mean="mean",
            ci95=lambda x: 1.96 * x.std(ddof=1) / np.sqrt(len(x))
        )
        .reset_index()
    )

    # Plot
    plt.figure(figsize=(12, 6))
    for cluster in raw_summary["cluster"].unique():
        sub = raw_summary[raw_summary["cluster"] == cluster]
        plt.errorbar(
            sub["measure"],
            sub["mean"],
            yerr=sub["ci95"],
            marker="o",
            capsize=5,
            label=f"Cluster {cluster}" if cluster != "LR" else "LR",
            color=colors.get(cluster, "black")
        )

    plt.xticks(rotation=45)
    plt.ylabel("Raw Score (BRIEF2 sign reversed)")
    plt.title(f"Behavioral Raw Scores (BRIEF2 sign reversed) by Cluster ({group} + LR)")
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)


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
    # Test if the clusters have significantly different number of HR+ and HR- subjects
    # Null hypotheseis: ASD outcome is not significantly different for the two clusters
    chi2, p, dof, expected = chi2_contingency(asds_by_cluster)
    print(f"Chi-square test: chi2 = {chi2:.2f}, p = {p:.3f}")

    # Fisher exact test (only works for 2 clusters)
    # Null hypothesis: ASD outcome is not significantly different for the two clusters
    # Gives odds ratio, which is probabiity that cluster 0 represents the HR+ group
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
    plt.show(block=False)

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

    # ================================
    # Histogram of Raw Scores (BRIEF2 flipped + Flanker + DCCS)
    # ================================

    # Colors for clusters + LR
    plot_colors = {
        "0": "#1f77b4",  # Blue
        "1": "#ff7f0e",  # Orange
        "LR": "#2ca02c"  # Green
    }

    # FacetGrid: one panel per measure
    g = sns.FacetGrid(
        df_raw_long,
        col="measure",
        col_wrap=3,
        sharex=False,
        sharey=False,
        height=4
    )

    # Plot histograms using hue="cluster" (all groups plotted together)
    g.map_dataframe(
        sns.histplot,
        x="raw_score",
        hue="cluster",
        palette=plot_colors,
        alpha=0.5,
        bins=20,
        stat="density"
    )

    # Axis labels and titles
    g.set_axis_labels("Raw Score", "Density")
    g.set_titles(col_template="{col_name}")

    # Add manual legend
    handles = [
        plt.Line2D([0], [0], color=clr, lw=8, alpha=0.5, label=lbl)
        for lbl, clr in zip(["Cluster 0", "Cluster 1", "LR"], ["#1f77b4", "#ff7f0e", "#2ca02c"])
    ]
    g.fig.legend(handles=handles, loc="upper right", title="Group", frameon=True)

    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(f"Distribution of Behavioral Raw Scores by Cluster (BRIEF2 sign reversed) ({group} + LR)")
    plt.show(block=False)

    return df_group