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
from plot_feature_histograms import plot_feature_histograms

target = "BRIEF2_GEC_T_score"
metric = 'cortical_thickness_VSA'
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


region_names_z = [name + "_z" for name in brain_cols]
plot_feature_histograms(df_hr_z, region_names_z)

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
X = df_hr_z[region_names_z]

results = []

for behavior in behavior_cols:
    y = behavior_df[behavior]
    mask = ~y.isna()
    X_sub, y_sub = X[mask], y[mask]

    # Elastic Net with cross-validated alpha and l1_ratio
    model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9], alphas=np.logspace(-3, 2, 10),
                         cv=5, max_iter=5000)

    scores = cross_val_score(model, X_sub, y_sub, cv=5, scoring='r2')

    # Fit on all data to get coefficients
    model.fit(X_sub, y_sub)

    # Get top 5 predictive regions by absolute weight
    coef_series = pd.Series(model.coef_, index=region_names_z)
    top_regions = coef_series.abs().sort_values(ascending=False).head(5)

    results.append({
        'Behavior': behavior,
        'Mean_R2_CV': np.mean(scores),
        'Top_Regions': top_regions.index.tolist(),
        'Top_Weights': top_regions.values.tolist()
    })

ml_df = pd.DataFrame(results)

# View results sorted by predictive power
print(ml_df.sort_values('Mean_R2_CV', ascending=False))

results = []

for behavior in behavior_cols:
    y = behavior_df[behavior]
    mask = ~y.isna()
    X_sub, y_sub = X[mask], y[mask]

    model = XGBRegressor(
        n_estimators=50,  # fewer trees
        max_depth=2,  # shallower trees
        learning_rate=0.05,  # slower learning
        subsample=0.8,  # random subset of rows
        colsample_bytree=0.8,  # random subset of features
        n_jobs=-1,
        random_state=42
    )

    scores = cross_val_score(model, X_sub, y_sub, cv=5, scoring='r2')
    model.fit(X_sub, y_sub)

    # Feature importance
    importance = pd.Series(model.feature_importances_, index=region_names_z)
    top_regions = importance.sort_values(ascending=False).head(5)

    results.append({
        'Behavior': behavior,
        'Mean_R2_CV': np.mean(scores),
        'Top_Regions': top_regions.index.tolist(),
        'Top_Importances': top_regions.values.tolist()
    })

ml_xgb_df = pd.DataFrame(results)

mystop=1

