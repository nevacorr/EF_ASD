import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, fisher_exact


def check_group_representation(df_all, ef_col, group_col='Group'):
    """
    Checks whether HR+ / HR- are represented differently across
    Low / Middle / High EF groups, prints the tables and tests,
    and draws a stacked bar chart.
    """
    d = df_all.copy()
    q_low, q_high = d[ef_col].quantile([0.25, 0.75])
    d['EF_group'] = np.where(d[ef_col] < q_low, 'Low',
                     np.where(d[ef_col] > q_high, 'High', 'Middle'))

    order = ['Low', 'Middle', 'High']

    # ---------------- Counts and percentages ----------------
    ct = pd.crosstab(d['EF_group'], d[group_col]).reindex(order)
    pct = ct.div(ct.sum(axis=1), axis=0) * 100

    print("Counts:")
    print(ct)
    print("\nRow percentages:")
    print(pct.round(1))

    # ---------------- Test: Low vs High (the groups the SVC saw) ----------------
    ct_ext = ct.loc[['Low', 'High']]
    chi2, p, dof, expected = chi2_contingency(ct_ext)
    print(f"\nLow vs High, chi-square: chi2 = {chi2:.2f}, p = {p:.3f}")

    if ct_ext.shape == (2, 2):
        odds, p_f = fisher_exact(ct_ext.values)
        print(f"Low vs High, Fisher exact: OR = {odds:.2f}, p = {p_f:.3f}")
        if (expected < 5).any():
            print("Some expected counts < 5: use the Fisher result.")

    # ---------------- EF by group ----------------
    print(f"\n{ef_col} by {group_col}:")
    print(d.groupby(group_col)[ef_col].agg(['count', 'mean', 'std']).round(2))

    # ---------------- Graphic ----------------
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = plt.cm.Set2(np.arange(len(pct.columns)))
    bottom = np.zeros(len(pct))

    for col, color in zip(pct.columns, colors):
        ax.bar(order, pct[col].values, bottom=bottom, label=col,
               color=color, edgecolor='white')
        for i, grp in enumerate(order):
            h = pct.loc[grp, col]
            ax.text(i, bottom[i] + h / 2, f"{h:.0f}%\n(n={ct.loc[grp, col]})",
                    ha='center', va='center', fontsize=9)
        bottom += pct[col].values

    ax.set_ylabel('% of children in EF group')
    ax.set_title(f'{group_col} composition by EF group')
    ax.legend(title=group_col, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    return ct, pct