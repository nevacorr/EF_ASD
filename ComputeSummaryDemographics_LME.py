import os
import pandas as pd
from Utility_Functions_Demographics import compute_stats_conditioned_on_identifiers
from Utility_Functions_Demographics import compute_stats_conditioned_on_identifiers_by_group
from Utility_Functions_Demographics import combine_redundant_columns, add_IQ_ADOS, add_race, add_missing_ages_from_brief2
from Utility_Functions_Demographics import convert_maternal_education_num_to_string, summarize_by_group

working_dir = os.getcwd()

R_directory = "/Users/nevao/R_Projects/IBIS_EF/processed_datafiles"
dataframes = {}

python_directory = "/Users/nevao/PycharmProjects/EF_ASD/"
python_filename = "IBIS_merged_df_full_addedmissing_ageschoolage_maternaled.csv"
VSA_ADOS_filename = "/Users/nevao/Documents/IBIS_EF/source data/Behav_Data/ADOS_school-age_long_data-2025-07-27T00_14_09.037Z.csv"
v24_v36_ADOS_filename = "/Users/nevao/Documents/IBIS_EF/source data/Behav_Data/ADOS_v24 and 36_long_data-2025-07-27T00_05_00.911Z.csv"
IQ_filename = "/Users/nevao/Documents/IBIS_EF/source data/Behav_Data/IQ data_Long_data-2025-07-26T23_51_14.152Z.csv"
V06_V12_race_filename = "/Users/nevao/Documents/IBIS_EF/source data/Behav_Data/race_v6_v12 from New spreadsheet.csv"
tsi_filename = "/Users/nevao/Documents/IBIS_EF/source data/Behav_Data/IBIS 1 and 2 TSI demographic_added_missing_data.csv"
nihtoolbox_race_filename = "/Users/nevao/Documents/IBIS_EF/source data/Behav_Data/NIH Toolbox_7-1-24_data-2024-07-01T19_40_36.204Z_addedmissingdata.csv"
brief2_ages_filename = "/Users/nevao/Documents/IBIS_EF/source data/Behav_Data/Missing_Ages_From_BRIEF2.csv"

dfs = []
ibis_df = None
brief2_ages = pd.read_csv(brief2_ages_filename)

file_list = ['ibis_subj_demographics_and_data_used_for_2025analysis_with_Brief2_subscales_with_brief1.csv',
                 'ab12_used_for_2025analysis__noIQ_ME.csv',
                 'ab24_used_for_2025analysis__noIQ_ME.csv',
                 'Flanker_Standard_Age_Corrected_used_for_2025analysis__noIQ_ME.csv',
                 'DCCS_Standard_Age_Corrected_used_for_2025analysis__noIQ_ME.csv',
                 'BRIEF2_GEC_T_score_used_for_2025analysis__noIQ_ME.csv',
                 ]


# Load all files
for filename in file_list:
    filepath = os.path.join(R_directory, filename)
    df_name = os.path.splitext(filename)[0]
    df_name = df_name.replace('_used_for_2025analysis_with_Brief2_subscales_with_brief1', '')
    df_name = df_name.replace('_used_for_2025analysis', '')
    df = pd.read_csv(filepath)

    if filename.lower().startswith("ibis"):
        ibis_df = df.copy()
    else:
        # Keep only Identifiers column and rename
        id_col = df[['Identifiers']].copy()
        id_col.rename(columns={'Identifiers': f'Identifiers_{df_name}'}, inplace=True)

        # Align with the ibis Identifiers
        merged = pd.merge(
            ibis_df[['Identifiers']],
            id_col,
            left_on='Identifiers',
            right_on=f'Identifiers_{df_name}',
            how='left'
        )

        # Create a new column matching ibis Identifiers, keep just the renamed one
        dfs.append(merged[[f'Identifiers_{df_name}']])

# Combine all identifier columns and full ibis data
id_df = pd.concat(dfs, axis=1)

# Merge school age columns
cols = [
    "Identifiers_Flanker_Standard_Age_Corrected__noIQ_ME",
    "Identifiers_BRIEF2_GEC_T_score__noIQ_ME",
    "Identifiers_DCCS_Standard_Age_Corrected__noIQ_ME"
]

# Combine across columns: take the first non-NaN value in the row
id_df["Identifiers_SchoolAge__noIQ_ME"] = id_df[cols].bfill(axis=1).iloc[:, 0]

# Remove flanker, dccs and brief2 columns
id_df.drop(columns=cols, inplace=True)

combined_df = pd.concat([ibis_df, id_df], axis=1)
substr_to_remove = ['Percent', 'T_score', 'raw_score', 'Standard_Age_Corrected', 'Risk', 'Combined_ASD_DX', 'mullen']
r_data_df = combined_df[(col for col in combined_df if not any(sub in col for sub in substr_to_remove))]

# Remove 'prefrontal_task' from column names
r_data_df.columns = [col.replace('prefrontal_task', '') for col in r_data_df.columns]

final_data = r_data_df.copy()
final_data.drop(columns=["X"], inplace=True)

# Add ADOS scores, IQ at 12mo, IQ at school age, age from DAS
final_data = add_IQ_ADOS(final_data, v24_v36_ADOS_filename,VSA_ADOS_filename,  IQ_filename)

final_data = add_race(final_data, V06_V12_race_filename, tsi_filename, nihtoolbox_race_filename)

final_data = add_missing_ages_from_brief2(final_data, brief2_ages)

final_data.to_csv(os.path.join(working_dir, 'demographics_by_subject.csv'))

demo_stats_ages_combined_by_group = summarize_by_group(final_data)

demo_stats = compute_stats_conditioned_on_identifiers(final_data, categorical_columns=[])

demo_stats_by_group = compute_stats_conditioned_on_identifiers_by_group(final_data)

demo_stats.to_csv(os.path.join(working_dir, 'demographic_stats_summary_with_DAS_age_race_with_brief1.csv'))

demo_stats_by_group.to_csv(os.path.join(working_dir, 'demographic_stats_summary_with_DAS_age_race_by_group_with_brief1.csv'))

mystop=1