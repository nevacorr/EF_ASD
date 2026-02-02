from create_input_for_ML import create_input_for_ML
from normative_model_brain_data import calc_normative_data
import os
import  numpy as np
from load_data_for_ML import load_all_data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

target = "BRIEF2_GEC_T_score"
metric = 'subcort_VSA'
#options 'volume_infant', 'volume_VSA', 'subcort_VSA', 'subcort_infant', 'ad_VSA', 'rd_VSA', 'md_VSA', 'fa_VSA'
#        'surface_area_VSA', 'cortical_thickness_VSA', 'subcort_infant+volume_infant'
working_dir = os.getcwd()

demographics = pd.read_csv(os.path.join(working_dir, 'demographics_by_subject.csv'), usecols=lambda c: c != "Unnamed: 0")
# Columns to drop explicitly
cols_to_drop = ['Sex', 'Group']

# Columns to drop based on substrings (case-insensitive)
substr_drop = ['Score', 'IQ', 'race', 'education']
cols_substr = [col for col in demographics.columns if any(s.lower() in col.lower() for s in substr_drop)]

# Combine
all_drop = cols_to_drop + cols_substr

demographics.drop(columns=all_drop, inplace=True)

df = load_all_data()

df, brain_cols, cov_cols = create_input_for_ML(df, metric)

cov_cols.remove('Site')
cov_cols.remove('Group')

df = pd.merge(demographics, df, on='Identifiers', how='outer')
df.drop(columns=['Identifiers'], inplace=True)

df = df[['CandID'] + [c for c in df.columns if c != 'CandID']]

ef_col = target

Age = "Final_Age_School_Age" if "BRIEF2" in target else None

cov_cols.append('Final_Age_School_Age')

df_hr_z = calc_normative_data(df, group_col='Group', lr_label='LR-', hr_labels=['HR+', 'HR-'],
                            brain_cols=brain_cols, ef_col=ef_col, covariates=cov_cols)

df_hr_z_nocandID = df_hr_z.drop(columns=['CandID'])

# Create behavior dataframe with same subjects

behavior_cols = ['CandID', 'BRIEF2_GEC_T_score', 'BRIEF2_GEC_raw_score',
                   'Flanker_Standard_Age_Corrected', 'DCCS_Standard_Age_Corrected']

# Filter df to only the rows in df_hr_z['CandID'] and only the columns in columns_to_keep
behavior_df = df[df['CandID'].isin(df_hr_z['CandID'])][behavior_cols]

# Reorder behavior_df to match the order of df_hr_z['CandID']
behavior_df = behavior_df.set_index('CandID').loc[df_hr_z['CandID']].reset_index()

# Per-region mean and SD
summary = df_hr_z_nocandID.agg(['mean', 'std'])
print(summary)

# Count of subjects with |z| > 2 (clinically significant deviation)
sig_counts = (df_hr_z_nocandID.abs() > 2).sum()
print("Subjects with |z|>2 per region:\n", sig_counts)

plt.figure(figsize=(12, 8))
region_names_z = [name + "_z" for name in brain_cols]
for i, region in enumerate(region_names_z):
    plt.subplot(3, 4, i+1)  # 3 rows x 4 columns for 12 regions
    sns.histplot(df_hr_z.loc[:,region], bins=20, kde=True, color='skyblue')
    plt.title(region)
    plt.xlabel("Z-score")
    plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Correlate with behavior
results = []

# Loop over each behavior and each brain region
behavior_cols.remove('CandID')
for behavior in behavior_cols:
    # Select the behavior column and corresponding CandID
    behavior_series = behavior_df[behavior]

    # Mask to exclude NaNs in this behavior
    not_nan_mask = ~behavior_series.isna()

    for region in region_names_z:
        # Only keep rows where behavior is not NaN
        region_values = df_hr_z.loc[not_nan_mask, region]
        behavior_values = behavior_series[not_nan_mask]

        rho, pval = pearsonr(region_values, behavior_values)
        results.append({
            'Behavior': behavior,
            'Region': region,
            'PearsonR': rho,
            'pval': pval
        })

# Convert to DataFrame
corr_df = pd.DataFrame(results)

# Filter for p-values < 0.1
significant = corr_df[corr_df['pval'] < 0.1]

# List all Region and Behavior pairs
sig_pairs = significant[['Region', 'Behavior']]

# Display as a list of tuples
sig_list = list(sig_pairs.itertuples(index=False, name=None))

print('signficant correlations')
print(sig_list)       # List of (Region, Behavior) tuples

# Features and behaviors
X = df_hr_z[region_names_z]  # signed z-scores
behavior_cols = [col for col in behavior_df.columns if col != 'CandID']

# Number of bootstrap repeats for stable feature importance
n_repeats = 50

results = []

for behavior in behavior_cols:
    y = behavior_df[behavior]

    # Exclude NaNs
    mask = ~y.isna()
    X_sub, y_sub = X.loc[mask], y[mask]

    # Store feature importances for bootstraps
    importance_all = pd.DataFrame(0, index=region_names_z, columns=range(n_repeats))

    for i in range(n_repeats):
        # Initialize XGBoost with conservative parameters for small dataset
        model = XGBRegressor(
            n_estimators=50,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42 + i
        )
        model.fit(X_sub, y_sub)
        importance_all[i] = model.feature_importances_

    # Average importance across bootstraps
    importance_mean = importance_all.mean(axis=1)
    importance_std = importance_all.std(axis=1)

    # Top 5 regions
    top_regions = importance_mean.sort_values(ascending=False).head(5)

    results.append({
        'Behavior': behavior,
        'Top_Regions': top_regions.index.tolist(),
        'Mean_Importance': top_regions.values.tolist(),
        'Std_Importance': importance_std[top_regions.index].tolist()
    })

# Convert to DataFrame for easy inspection
ml_xgb_df = pd.DataFrame(results)

# Optional: sort by first top importance for visualization
ml_xgb_df['Max_Importance'] = ml_xgb_df['Mean_Importance'].apply(max)
ml_xgb_df = ml_xgb_df.sort_values('Max_Importance', ascending=False).drop(columns='Max_Importance')

mystop=1

