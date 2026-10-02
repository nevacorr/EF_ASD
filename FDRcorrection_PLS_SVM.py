from statsmodels.stats.multitest import multipletests

p_values = [p_flanker, p_dccs, p_third]

reject, p_corrected, _, _ = multipletests(
    p_values,
    alpha=0.05,
    method='fdr_bh'
)

print("Original p-values:", p_values)
print("FDR-corrected p-values:", p_corrected)
print("Significant after FDR:", reject)