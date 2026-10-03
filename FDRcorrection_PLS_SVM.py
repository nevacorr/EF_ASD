from statsmodels.stats.multitest import multipletests

p_values_pls_da = [0.005, 0.477, 0.302]  #[p_flanker, p_dccs, p_BRIEF]
AUC_pls_da = [0.844, 0.519, 0.568]
p_values_svm = [0.006, 0.489, 0.073]
SVM_AUC_svm = [0.832, 0.513, 0.674]

for p_values in [p_values_pls_da, p_values_svm]:
    reject, p_corrected, _, _ = multipletests(
        p_values,
        alpha=0.05,
        method='fdr_bh'
    )

    print("Original p-values:", p_values)
    print("FDR-corrected p-values:", p_corrected)
    print("Significant after FDR:", reject)