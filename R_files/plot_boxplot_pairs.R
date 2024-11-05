
# plot_boxplot_pairs.R

plot_boxplot_pairs <- function(df, col1, col2, title) {

  # Create a long format dataframe  for plotting on same figure
  long_df <- df %>%
    pivot_longer(cols = c(!!sym(col1), !!sym(col2)),
                 names_to = "Score_Type", 
                 values_to = "Score_Value")
  
  # Create boxplots
  ggplot(long_df, aes(x = Risk, y = Score_Value, fill = Score_Type)) +
    geom_boxplot(na.rm = TRUE) +
    scale_fill_manual(values = c("lightblue", "lightgreen")) +
    labs(title = title, 
       x = "Risk",
       y = title) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))  # Center the title
  }