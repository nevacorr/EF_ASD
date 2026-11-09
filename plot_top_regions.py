import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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