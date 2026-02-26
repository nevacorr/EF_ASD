import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy.stats import pearsonr

def pls_da(final_brain_df, brain_cols, df_hr, ef_col, perform_norm_modeling):
    import pandas as pd
    # -----------------------------
    # Step 1: Prepare your data
    # -----------------------------
    # X_brain: subjects x brain features (numpy array or DataFrame)
    # y_EF: continuous EF scores (numpy array or Series)

    # Example placeholder data
    # X_brain = pd.DataFrame(np.random.rand(50, 100))  # 50 subjects, 100 brain features
    # y_EF = pd.Series(np.random.rand(50))

    if perform_norm_modeling:
        brain_cols= [col + '_z' for col in brain_cols]

    df_ef=final_brain_df[['Identifiers', ef_col, 'Group']].copy()
    df_all=pd.merge(df_ef, df_hr,  on="Identifiers", how="inner")
    df_all=df_all.dropna().reset_index(drop=True)

    X_brain = df_all[brain_cols].copy()
    y_EF = df_all[ef_col].copy()
    X_Group = df_all['Group'].copy()

    # -----------------------------
    # Step 2: Extreme group selection
    # -----------------------------
    q_low = y_EF.quantile(0.25)
    q_high = y_EF.quantile(0.75)
    mask = (y_EF <= q_low) | (y_EF >= q_high)

    X_group = X_brain[mask]
    y_group = y_EF[mask].copy()

    # Binary labels: 0 = Low EF, 1 = High EF
    y_group[:] = (y_group > q_high).astype(int)

    # -----------------------------
    # Step 4: Standardize features
    # -----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_group)

    # -----------------------------
    # Step 5: Train/Test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_group, test_size=0.2, stratify=y_group, random_state=42
    )

    # -----------------------------
    # Step 4: Cross-validation to select n_components
    # -----------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    max_components = min(X_train.shape[0], X_train.shape[1], 5)

    best_auc = 0
    best_n = 1

    for n_comp in range(1, max_components + 1):
        pls = PLSRegression(n_components=n_comp)
        aucs = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            pls.fit(X_train[train_idx], y_train.iloc[train_idx])
            y_val_pred = pls.predict(X_train[val_idx]).ravel()
            aucs.append(roc_auc_score(y_train.iloc[val_idx], y_val_pred))
        mean_auc = np.mean(aucs)
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_n = n_comp

    print(f"Selected n_components: {best_n}, CV AUC: {best_auc:.3f}")

    # -----------------------------
    # Step 5: Train final model on full training set
    # -----------------------------
    pls_final = PLSRegression(n_components=best_n)
    pls_final.fit(X_train, y_train)

    # Predict on test set
    y_test_pred = pls_final.predict(X_test).ravel()
    test_auc = roc_auc_score(y_test, y_test_pred)
    print(f"Test ROC AUC: {test_auc:.3f}")

    # -----------------------------
    # Step 6: Permutation testing on test set
    # -----------------------------
    n_permutations = 1000
    perm_aucs = []

    for i in tqdm(range(n_permutations)):
        y_perm = np.random.permutation(y_test)  # shuffle test labels
        perm_auc = roc_auc_score(y_perm, y_test_pred)
        perm_aucs.append(perm_auc)

    perm_aucs = np.array(perm_aucs)
    p_value = (np.sum(perm_aucs >= test_auc) + 1) / (n_permutations + 1)
    print(f"Permutation p-value for test AUC: {p_value:.3f}")

    # -----------------------------
    # Step 7: Feature importance
    # -----------------------------
    feature_weights = pls_final.x_weights_[:, 0]
    brain_feature_importance = pd.Series(feature_weights, index=X_brain.columns).sort_values(ascending=False)
    print("Top 10 features driving Low vs High EF separation:")
    print(brain_feature_importance.head(10))

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import f_oneway
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    # -----------------------------
    # Step 1: Compute Frontal EF Index
    # -----------------------------
    # Use your 4 PLS features and weights
    frontal_index = (
            0.746488 * X_brain['Frontal_L_WM_VSA'] +
            0.467038 * X_brain['Frontal_L_GM_VSA'] +
            0.452255 * X_brain['Frontal_R_WM_VSA'] +
            -0.141765 * X_brain['Frontal_R_GM_VSA']
    )

    # Create DataFrame including risk group
    df_plot = pd.DataFrame({
        'Frontal_Index': frontal_index,
        'Risk_Group': X_Group  # HR+, HR-, LR-
    })

    # -----------------------------
    # Step 2: Boxplot with swarm
    # -----------------------------
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Risk_Group', y='Frontal_Index', data=df_plot, palette="pastel")
    sns.swarmplot(x='Risk_Group', y='Frontal_Index', data=df_plot, color=".25")
    plt.ylabel('Frontal EF Index (PLS Component 1)')
    plt.title('Frontal EF Index by Risk Group')
    plt.show()

    # -----------------------------
    # Step 3: Statistical testing (ANOVA)
    # -----------------------------
    HR_plus = df_plot[df_plot['Risk_Group'] == 'HR+']['Frontal_Index']
    HR_minus = df_plot[df_plot['Risk_Group'] == 'HR-']['Frontal_Index']
    LR_minus = df_plot[df_plot['Risk_Group'] == 'LR-']['Frontal_Index']

    F_stat, p_val = f_oneway(HR_plus, HR_minus, LR_minus)
    print(f"ANOVA F={F_stat:.2f}, p={p_val:.3f}")

    # -----------------------------
    # Step 4: Post-hoc comparisons (Tukey HSD)
    # -----------------------------
    tukey = pairwise_tukeyhsd(endog=df_plot['Frontal_Index'],
                              groups=df_plot['Risk_Group'],
                              alpha=0.05)
    print(tukey)

    # Optional: plot Tukey HSD results
    tukey.plot_simultaneous()
    plt.title("Tukey HSD: Pairwise Risk Group Comparisons")
    plt.show()

    # -----------------------------
    # Step 1: Add EF scores to the plotting DataFrame
    # -----------------------------
    df_plot['EF_Score'] = df_all[ef_col]  # make sure df_all has EF scores
    df_plot['Frontal_Index'] = frontal_index  # your PLS-weighted index

    # -----------------------------
    # Step 2: Overall correlation
    # -----------------------------
    r_all, p_all = pearsonr(df_plot['Frontal_Index'], df_plot['EF_Score'])
    print(f"Overall correlation: r = {r_all:.3f}, p = {p_all:.3f}")

    # Scatter plot with regression line
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x='Frontal_Index', y='EF_Score', hue='Risk_Group', data=df_plot, s=70)
    sns.regplot(x='Frontal_Index', y='EF_Score', data=df_plot, scatter=False, color='gray')
    plt.title(f'Frontal EF Index vs EF Score (Overall) \nr={r_all:.2f}, p={p_all:.3f}')
    plt.xlabel('Frontal EF Index (PLS component 1)')
    plt.ylabel('EF Score')
    plt.legend(title='Risk Group')
    plt.show()

    # -----------------------------
    # Step 3: Within-group correlations
    # -----------------------------
    for group in df_plot['Risk_Group'].unique():
        sub = df_plot[df_plot['Risk_Group'] == group]
        r, p = pearsonr(sub['Frontal_Index'], sub['EF_Score'])
        print(f"{group}: r = {r:.3f}, p = {p:.3f}")

        # Optional: scatter plot per group
        plt.scatter(sub['Frontal_Index'], sub['EF_Score'], label=f"{group} (r={r:.2f})")

    plt.xlabel('Frontal EF Index (PLS component 1)')
    plt.ylabel('EF Score')
    plt.title('Frontal EF Index vs EF Score by Risk Group')
    plt.legend()
    plt.show()

    # -----------------------------
    # Step 1: Define Low vs High EF groups
    # -----------------------------
    # You can define extremes based on percentiles, e.g., bottom/top 25%
    low_thresh = df_plot['EF_Score'].quantile(0.25)
    high_thresh = df_plot['EF_Score'].quantile(0.75)

    df_plot['EF_Group'] = 'Middle'
    df_plot.loc[df_plot['EF_Score'] <= low_thresh, 'EF_Group'] = 'Low EF'
    df_plot.loc[df_plot['EF_Score'] >= high_thresh, 'EF_Group'] = 'High EF'

    # Keep only Low and High for plotting
    df_extremes = df_plot[df_plot['EF_Group'].isin(['Low EF', 'High EF'])]

    # -----------------------------
    # Step 2: Boxplot with swarm for extremes
    # -----------------------------
    plt.figure(figsize=(6, 5))
    sns.boxplot(x='EF_Group', y='Frontal_Index', data=df_extremes, palette="pastel")
    sns.swarmplot(x='EF_Group', y='Frontal_Index', data=df_extremes, color=".25")
    plt.ylabel('Frontal EF Index (PLS Component 1)')
    plt.title('Frontal EF Index: Low vs High EF')
    plt.show()

    # -----------------------------
    # Step 3: Statistical test (t-test)
    # -----------------------------
    from scipy.stats import ttest_ind

    low_vals = df_extremes[df_extremes['EF_Group'] == 'Low EF']['Frontal_Index']
    high_vals = df_extremes[df_extremes['EF_Group'] == 'High EF']['Frontal_Index']

    t_stat, p_val = ttest_ind(low_vals, high_vals)
    print(f"Low vs High EF: t = {t_stat:.3f}, p = {p_val:.3f}")