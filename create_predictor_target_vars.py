from Utility_Functions_XGBoost import plot_correlations, remove_collinearity
from matplotlib import pyplot as plt
def create_predictor_target_vars(dforig, target, metric, include_group, run_dummy_quick_fit_xgb,
                                 show_heat_map, remove_colinear):

    df = dforig.copy()

    # Extract first three characters of 'Identifiers' and create new column 'Site''
    df['Site'] = df['Identifiers'].str[:3]

    # Move 'Site' to be the third column
    site_col = df.pop('Site')
    df.insert(2, 'Site', site_col)

    df.drop(columns=['Identifiers'], inplace=True)

    # Select columns of interest based on metric
    if metric == "volume_infant":
        pred_brain_cols = df.columns[df.columns.str.contains("WM_V12|WM_V24|GM_V12|GM_V24", regex=True)].tolist()
    elif metric == "volume_VSA":
        pred_brain_cols = df.columns[df.columns.str.contains("WM_VSA|GM_VSA", regex=True)].tolist()
    elif metric == "subcort_infant":
        pred_brain_cols = df.columns[df.columns.str.contains(
            "Amygdala_v12|Putamen_v12|Caudate_v12|Thalamus_v12|GlobusPall_v12|Hippocampus_v12|Amygdala_v24|Putamen_v24|Caudate_v24|Thalamus_v24|GlobusPall_v24|Hippocampus_v24",
            regex=True)].tolist()
    elif metric == "subcort_infant+volume_infant":
        pred_brain_cols = df.columns[
            df.columns.str.contains(
                (
                    "Amygdala_v12|Putamen_v12|Caudate_v12|Thalamus_v12|GlobusPall_v12|Hippocampus_v12|"
                    "Amygdala_v24|Putamen_v24|Caudate_v24|Thalamus_v24|GlobusPall_v24|Hippocampus_v24|"
                    "WM_V12|WM_V24|GM_V12|GM_V24"
                ),
                regex=True
            )
        ].tolist()
    elif metric == "subcort_VSA":
        pred_brain_cols = df.columns[df.columns.str.contains(
            r"(?:Amygdala|Caudate|Putamen|Thalamus|Hippocampus|Globus_Pall).*VSA",regex=True)].tolist()
    elif metric == "cortical_thickness_VSA":
        pred_brain_cols = df.columns[df.columns.str.contains("CT_VSA")].tolist()
    elif metric == "surface_area_VSA":
        pred_brain_cols = df.columns[df.columns.str.contains("SA_VSA")].tolist()
    elif metric in {"fa_VSA", "rd_VSA", "md_VSA", "ad_VSA"}:
        prefix = metric.split('_')[0]  # Extracts the prefix (e.g., "fa" from "fa_VSA")
        pred_brain_cols = df.columns[df.columns.str.startswith(prefix.upper() + '_')].tolist()

    if include_group:
        pred_non_brain_cols = ["Site", "Sex", "Group", "Group_HR+", "Group_HR-", "Group_LR-"]
    else:
        pred_non_brain_cols = ["Site", "Sex", "Group"]

    predictor_list = pred_non_brain_cols + pred_brain_cols

    # Keep only rows where the target variable is not NA
    df = df[df[target].notna()]

    if run_dummy_quick_fit_xgb == 1:
        df = df.sample(frac=0.1, random_state=42)
        n_iter = 5

    # Make variable with just group
    group_vals = df['Group'].copy()
    group_vals = group_vals.reset_index(drop=True)

    # Make variable with just sex
    sex_vals = df['Sex'].copy()
    sex_vals = sex_vals.reset_index(drop=True)

    # Make matrix of predictors
    X = df[predictor_list].copy()
    X.drop(columns=['Group'], inplace=True)

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

    return X, y, group_vals, sex_vals
