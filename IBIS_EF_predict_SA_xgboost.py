import pandas as pd
import os
from pycaret.regression import setup, create_model, tune_model, finalize_model, predict_model, save_model, load_model
import seaborn as sns
import matplotlib.pyplot as plt
from Utility_Functions_XGBoost import load_and_clean_data, plot_correlations

target = "Flanker_Standard_Age_Corrected"
run_training = 1

working_dir = os.getcwd()

datafilename = "final_df_for_xgboost.csv"

# load and clean data
df = load_and_clean_data(working_dir, datafilename, target)

# plot feature correlation heatmap

plot_title="Correlation between regional volume measures after dividing by totTissue"
corr_matrix = plot_correlations(df, target, plot_title)

plt.show()

if run_training:

    # Set up 10-fold cross validation
    exp = setup(
        data=df,
        target=target,
        fold=10
    )

    # Tune XGBoost model with Bayesian optimization
    xgb = create_model("xgboost")

    # Tuning the model using Bayesian optimization with Optuna
    tuned_xgb = tune_model(
        xgb,
        optimize="R2",
        search_library = "optuna",
        search_algorithm="tpe",
        n_iter=50,
        early_stopping = True
    )

    # Finalize the tuned model (train it on the full dataset)
    final_model = finalize_model(tuned_xgb)

    # Predict on the full dataset using the finalized model
    predictions = predict_model(final_model, data=df)

    save_model(final_model, f"{target}_final_xgboost_model")
    predictions.to_csv(f"{target}_predictions.csv")

else:

    predictions = pd.read_csv(f"{target}_predictions.csv")
    final_model = load_model(f"{target}_final_xgboost_model")

# Plot actual vs. predicted values
plt.figure(figsize=(10,8))
sns.scatterplot(x=predictions[target] , y=predictions['prediction_label'])

# plot y = x line
plt.plot([min(predictions[target]), max(predictions[target])],
         [min(predictions[target]), max(predictions[target])],
         linestyle = "--", color="red")

plt.xlabel(f"Actual {target}")
plt.ylabel(f"Predicted {target}")
plt.title(f"Predicted vs. Actual {target}")
plt.show()

mystop = 1