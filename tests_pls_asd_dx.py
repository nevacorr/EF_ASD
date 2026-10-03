from scipy.stats import mannwhitneyu
import pandas as pd
from statsmodels.stats.multitest import multipletests

def test_pls_asd_dx(pls_final, X_all_scaled, df_all, mask):

    pls_scores = pls_final.transform(X_all_scaled)

    score_df = pd.DataFrame({
        'Identifiers': df_all.loc[mask, 'Identifiers'].values,
        'PLS_Component_1': pls_scores[:, 0],
        'PLS_Component_2': pls_scores[:, 1]
    })

    score_df = score_df.merge(
        df_all[['Identifiers', 'Group']],
        on='Identifiers',
        how='left'
    )

    p_values = []

    for component in ['PLS_Component_1', 'PLS_Component_2']:

        autism = score_df.loc[
            score_df['Group'] == "HR+",
            component
        ].dropna()

        no_autism = score_df.loc[
            score_df['Group'] == "HR-",
            component
        ].dropna()

        statistic, p_value = mannwhitneyu(
            autism,
            no_autism,
            alternative='two-sided'
        )

        p_values.append(p_value)

        print(f"\n{component}")
        print(f"HR+: N={len(autism)}, median={autism.median():.3f}")
        print(f"HR-: N={len(no_autism)}, median={no_autism.median():.3f}")
        print(f"Mann-Whitney U = {statistic:.3f}")
        print(f"p uncorrected = {p_value:.4f}")

    # FDR correction across the two PLS components
    p_corrected = multipletests(
        p_values,
        method='fdr_bh'
    )[1]

    print("\nFDR-corrected p-values:")
    print(f"Component 1: {p_corrected[0]:.4f}")
    print(f"Component 2: {p_corrected[1]:.4f}")

    return score_df