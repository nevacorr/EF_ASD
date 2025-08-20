import os
import pandas as pd
from Utility_Functions_Demographics import compute_stats_conditioned_on_identifiers
from Utility_Functions_Demographics import combine_redundant_columns, add_IQ_ADOS
from Utility_Functions_Demographics import convert_maternal_education_num_to_string

working_dir = os.getcwd()

R_directory = "/Users/nevao/R_Projects/IBIS_EF/processed_datafiles"
dataframes = {}

python_directory = "/Users/nevao/PycharmProjects/EF_ASD/"
python_filename = "IBIS_merged_df_full_demographics.csv"
VSA_ADOS_filename = "/Users/nevao/Documents/IBIS_EF/source data/Behav_Data/ADOS_school-age_long_data-2025-07-27T00_14_09.037Z.csv"
v24_v36_ADOS_filename = "/Users/nevao/Documents/IBIS_EF/source data/Behav_Data/ADOS_v24 and 36_long_data-2025-07-27T00_05_00.911Z.csv"
IQ_filename = "/Users/nevao/Documents/IBIS_EF/source data/Behav_Data/IQ data_Long_data-2025-07-26T23_51_14.152Z.csv"

dfs = []
ibis_df = None

# Load all files
for filename in os.listdir(R_directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(R_directory, filename)
        df_name = os.path.splitext(filename)[0]
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
combined_df = pd.concat([ibis_df, id_df], axis=1)
substr_to_remove = ['Percent', 'T_score', 'raw_score', 'Standard_Age_Corrected', 'Risk', 'Combined_ASD_DX']
r_data_df = combined_df[(col for col in combined_df if not any(sub in col for sub in substr_to_remove))]

# Remove 'prefrontal_task' from column names
r_data_df.columns = [col.replace('prefrontal_task', '') for col in r_data_df.columns]

python_data = pd.read_csv(os.path.join(python_directory, python_filename))

python_data.drop(columns=['DoB', 'VSD-All NIHToolBox,Date_taken', 'Risk', 'Sex',
                        'V06 tsi,father_education','VSD-All demographics,ASD_Ever_DSMIV',
                        'VSD-All NIHToolBox,Registration_Data_Fathers_Education',
                        'VSD-All NIHToolBox,Registration_Data_Guardians_Education',
                        'VSD-All NIHToolBox,Registration_Data_Education',
                        'V06 tsi,child_ethnicity','V06 tsi,candidate_race'], inplace=True)

vsa_me_col_name = 'VSD-All NIHToolBox,Registration_Data_Mothers_Education'
python_data[vsa_me_col_name] = python_data[vsa_me_col_name].apply(convert_maternal_education_num_to_string)

column_groups = [['V06 demographics,candidate_ethnicity', 'V12 demographics,candidate_ethnicity'],
    ['V06 demographics,candidate_race', 'V12 demographics,candidate_race'],
    ['V06 tsi,mother_education', 'VSD-All NIHToolBox,Registration_Data_Mothers_Education']]

new_column_names = ['V06V12candidate_ethnicity', 'V06V12candidate_race', 'AllAges_MotherEducation']

python_data = combine_redundant_columns(python_data, column_groups, new_column_names)

# column_groups = [['V06V12candidate_ethnicity', 'VSD-All NIHToolBox,Registration_Data_Ethnicity'],
#                  ['V06V12candidate_race', 'VSD-All NIHToolBox,Registration_Data_Race']]
#
# new_column_names = ['AllAges_Ethnicity', 'AllAges_Race']
#
# python_data = combine_redundant_columns(python_data, column_groups, new_column_names)

final_data = pd.merge(r_data_df, python_data, on='Identifiers', how='outer')
final_data.drop(columns=["X"], inplace=True)

# Add ADOS scores, IQ at 12mo, IQ at school age
final_data = add_IQ_ADOS(final_data, v24_v36_ADOS_filename,VSA_ADOS_filename,  IQ_filename)

demo_stats = compute_stats_conditioned_on_identifiers(final_data, categorical_columns=[])

demo_stats.to_csv(os.path.join(working_dir, 'demographic_stats_summary.csv'))

mystop=1