from Utility_Functions_XGBoost import plot_correlations, remove_collinearity
from matplotlib import pyplot as plt
def create_predictor_target_vars(df, target, metric, include_group, run_dummy_quick_fit_xgb,
                                 show_heat_map, remove_colinear):

    # Extract first three characters of 'Identifiers' and create new column 'Site''
    df['Site'] = df['Identifiers'].str[:3]

    # Move 'Site' to be the third column
    site_col = df.pop('Site')
    df.insert(2, 'Site', site_col)

    df.drop(columns=['Identifiers'], inplace=True)

    df = df.loc[:, ~df.columns.str.contains('ICV|totTiss', regex=True)]

    # Select columns of interest based on metric
    if metric == "volume_infant":
        pred_brain_cols = df.columns[df.columns.str.contains("WM|GM", regex=True)]
    elif metric == "subcort_infant":
        pred_brain_cols = df.columns[df.columns.str.contains(
            "Amygdala|Putamen|Caudate|Thalamus|GlobusPall|Hippocampus",
            case=False, regex=True
        )]
    elif metric in {"fa_VSA", "rd_VSA", "md_VSA", "ad_VSA"}:
        prefix = metric.split('_')[0]  # Extracts the prefix (e.g., "fa" from "fa_VSA")
        pred_brain_cols = df.columns[df.columns.str.startswith(prefix.upper() + '_')]

    if include_group:
        pred_non_brain_cols = ["Site", "Sex","Group_HR+", "Group_HR-", "Group_LR-"]
    else:
        pred_non_brain_cols = ["Site", "Sex"]

    predictor_list = pred_non_brain_cols + pred_brain_cols

    # Keep only rows where the target variable is not NA
    df = df[df[target].notna()]

    if run_dummy_quick_fit_xgb == 1:
        df = df.sample(frac=0.1, random_state=42)
        n_iter = 5

    # Make matrix of predictors
    X = df[predictor_list].copy()

    # Make vector with target value
    y = df[target].values

    if show_heat_map:
        # plot feature correlation heatmap
        plot_title = f"Correlation between regional {metric}"
        corr_matrix = plot_correlations(X, plot_title)
        plt.show()

    if remove_colinear:
        # remove features so that none have more than 0.9 correlation with other
        df = remove_collinearity(X, 0.9)
        plot_title="After removing colinear features"
        corr_matrix = plot_correlations(X, plot_title)
        plt.show()

    return X, y
