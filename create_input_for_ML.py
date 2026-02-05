from Utility_Functions_XGBoost import plot_correlations, remove_collinearity
from matplotlib import pyplot as plt
def create_input_for_ML(dforig, metric):

    df = dforig.copy()
    ICV_cols = []

    # Extract first three characters of 'Identifiers' and create new column 'Site''
    df['Site'] = df['Identifiers'].str[:3]

    # Move 'Site' to be the third column
    site_col = df.pop('Site')
    df.insert(2, 'Site', site_col)

    # Select columns of interest based on metric
    if metric == "volume_infant":
        pred_brain_cols = df.columns[df.columns.str.contains("WM_V12|WM_V24|GM_V12|GM_V24", regex=True)].tolist()
        ICV_cols=['ICV_V12', 'ICV_V24']
    elif metric == "volume_VSA":
        pred_brain_cols = df.columns[df.columns.str.contains("WM_VSA|GM_VSA", regex=True)].tolist()
        ICV_cols=['ICV_vol_VSA']
    elif metric == "subcort_infant":
        pred_brain_cols = df.columns[df.columns.str.contains(
            "Amygdala_v12|Putamen_v12|Caudate_v12|Thalamus_v12|GlobusPall_v12|Hippocampus_v12|Amygdala_v24|Putamen_v24|Caudate_v24|Thalamus_v24|GlobusPall_v24|Hippocampus_v24",
            regex=True)].tolist()
        ICV_cols = ['ICV_V12', 'ICV_V24']
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
        pred_brain_cols.remove('ICV_V12')
        pred_brain_cols.remove('ICV_V24')
        ICV_cols = ['ICV_V12', 'ICV_V24', 'ICV_vol_VSA']
    elif metric == "subcort_VSA":
        pred_brain_cols = df.columns[df.columns.str.contains(
            r"(?:Amygdala|Caudate|Putamen|Thalamus|Hippocampus|Globus_Pall).*VSA",regex=True)].tolist()
        ICV_cols=['ICV_vol_VSA']
    elif metric == "subcort_infant + subcort_VSA":
        pred_brain_cols = df.columns[
            df.columns.str.contains(
                r"(Amygdala|Putamen|Caudate|Thalamus|GlobusPall|Globus_Pall|Hippocampus)"
                r".*(v12|v24|VSA)",
                regex=True
            )
        ].tolist()
        ICV_cols = ['ICV_V12', 'ICV_V24', 'ICV_vol_VSA']
    elif metric == "cortical_thickness_VSA":
        pred_brain_cols = df.columns[df.columns.str.contains("CT_VSA")].tolist()
    elif metric == "surface_area_VSA":
        pred_brain_cols = df.columns[df.columns.str.contains("SA_VSA")].tolist()
    elif metric in {"fa_VSA", "rd_VSA", "md_VSA", "ad_VSA"}:
        prefix = metric.split('_')[0]  # Extracts the prefix (e.g., "fa" from "fa_VSA")
        pred_brain_cols = df.columns[df.columns.str.startswith(prefix.upper() + '_')].tolist()

    pred_non_brain_cols = ["Site", "Sex", "Group"] + ICV_cols

    return df, pred_brain_cols, pred_non_brain_cols
