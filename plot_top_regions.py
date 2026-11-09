import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_top_regions(df_all, importance_df, ef_col, n_top=6):
    d = df_all.copy()
    q_low, q_high = d[ef_col].quantile([0.25, 0.75])
    d['EF_group'] = np.where(d[ef_col] < q_low, 'Low',
                     np.where(d[ef_col] > q_high, 'High', 'Middle'))

    top = importance_df.head(n_top)['Region'].tolist()
    long = d.melt(id_vars=['EF_group'], value_vars=top,
                  var_name='Region', value_name='z')

    order = ['Low', 'Middle', 'High']
    g = sns.catplot(data=long, x='EF_group', y='z', col='Region',
                    col_wrap=3, order=order, kind='box',
                    showfliers=False, sharey=False, height=3.5)
    g.map_dataframe(sns.stripplot, x='EF_group', y='z', order=order,
                    color='k', alpha=0.35, size=3)
    for ax in g.axes.flat:
        ax.axhline(0, ls='--', c='gray', lw=1)
    plt.tight_layout()
    plt.show()
    mystop=1

def plot_top_regions_fine(df_all, importance_df, ef_col, n_top=6):
    d = df_all.copy()

    # Low (0-25%), middle split into 5 slices, High (75-100%)
    edges = [0, .25, .42, .58, .75, 1.0]
    order = ['Low', 'M1', 'M2', 'M3', 'High']
    d['EF_group'] = pd.qcut(d[ef_col].rank(method='first'), q=edges, labels=order).astype(str)

    top = importance_df.head(n_top)['Region'].tolist()
    long = d.melt(id_vars=['EF_group'], value_vars=top,
                  var_name='Region', value_name='z')

    g = sns.catplot(data=long, x='EF_group', y='z', col='Region',
                    col_wrap=3, order=order, kind='box',
                    showfliers=False, sharey=False, height=3.5, aspect=1.4)
    g.map_dataframe(sns.stripplot, x='EF_group', y='z', order=order,
                    color='k', alpha=0.35, size=3)
    for ax in g.axes.flat:
        ax.axhline(0, ls='--', c='gray', lw=1)
    plt.tight_layout()
    plt.show()
    mystop=1

def plot_top_regions_mean_ci(df_all, importance_df, ef_col, n_top=9,
                             edges=None, order=None, show_points=False):
    if edges is None:
        edges = [0, .25, .42, .58, .75, 1.0]
        order = ['Low', 'M1', 'M2', 'M3', 'High']

    d = df_all.copy()
    d['EF_group'] = pd.qcut(d[ef_col].rank(method='first'),
                            q=edges, labels=order).astype(str)

    top = importance_df.head(n_top)['Region'].tolist()
    ncols = 3
    nrows = int(np.ceil(len(top) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows),
                             sharex=True, squeeze=False)

    for ax, reg in zip(axes.flat, top):
        g = d.groupby('EF_group')[reg].agg(['mean', 'sem', 'count']).reindex(order)
        x = np.arange(len(order))

        if show_points:
            for i, grp in enumerate(order):
                vals = d.loc[d['EF_group'] == grp, reg]
                ax.scatter(np.full(len(vals), i) + np.random.uniform(-0.12, 0.12, len(vals)),
                           vals, color='gray', alpha=0.25, s=8)

        ax.errorbar(x, g['mean'], yerr=1.96 * g['sem'],
                    marker='o', capsize=4, color='C0', zorder=3)
        ax.axhline(0, ls='--', c='gray', lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{o}\n(n={int(n)})" for o, n in zip(order, g['count'])],
                           fontsize=8)
        ax.set_title(reg, fontsize=9)
        ax.set_ylabel('Mean z (95% CI)')

    for ax in axes.flat[len(top):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()