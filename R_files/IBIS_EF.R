
# install.packages("ggplot2")
# install.packages("tidyr")

# library(ggplot2)
# library(tidyr)

rm(list = ls())

source("./plot_boxplot_pairs.R")

ibis_behav <- read.csv(file.path("/Users/neva/Documents/IBIS Executive Function/dataframes and plots/",
                                  "IBIS_behav_dataframe_demographics_AnotB_Flanker_DCCS.csv"))

# Clean the Risk column: Remove leading/trailing spaces and convert empty strings to NA
ibis_behav$Risk <- trimws(ibis_behav$Risk)  # Remove spaces
ibis_behav$Risk[ibis_behav$Risk == ""] <- NA  # Convert empty strings to NA

# Filter rows
ibis_behav_filtered <- ibis_behav[!is.na(ibis_behav$Risk), ]

# Check for NA values in your key columns
sum(is.na(ibis_behav_filtered$Score_Value))
sum(is.na(ibis_behav_filtered$Risk))

# plot_boxplot_pairs(ibis_behav_filtered, "AB_12_Percent", "AB_24_Percent", "AB Scores Box Plots")

# plot_boxplot_pairs(ibis_behav_filtered, "AB_Reversals_12_Percent", "AB_Reversals_24_Percent", 
                   "AB Reversal Box Plots")

plot_boxplot_pairs(ibis_behav_filtered, "Flanker_Standard_Age_Corrected", "DCCS_Standard_Age_Corrected", 
                   "School Age Executive Function Measure Box Plots")

