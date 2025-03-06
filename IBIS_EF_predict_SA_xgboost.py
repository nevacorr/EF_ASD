import pandas as pd
import os
from pycaret.regression import setup, create_model, tune_model, finalize_model, predict_model, save_model, load_model
import matplotlib.pyplot as plt
import seaborn as sns

output_to_predict = "Flanker_Standard_Age_Corrected"
run_training = 1

if run_training:

    working_dir = os.getcwd()

    df = pd.read_csv(f"{working_dir}/final_df_for_xgboost.csv")

    columns_to_exclude = ["CandID", "Identifiers", "Combined_ASD_DX", "Risk", "Group", "AB_12_Percent", "AB_24_Percent",
                       "BRIEF2_GEC_raw_score", "BRIEF2_GEC_T_score", "DCCS_Standard_Age_Corrected"]

    df.drop(columns=columns_to_exclude, inplace=True)

    # Keep only rows where the response variable is not NA
    df = df[df[output_to_predict].notna()]

    #Encode Sex column Female = 0 Male = 1
    df["Sex"] = df["Sex"].replace({"Female": 0, "Male": 1})

    # Set up 10-fold cross validation
    exp = setup(
        data=df,
        target=output_to_predict,

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
        early_stopping = True,
        n_jobs=-1
    )

    # Finalize the tuned model (train it on the full dataset)
    final_model = finalize_model(tuned_xgb)

    # Predict on the full dataset using the finalized model
    predictions = predict_model(final_model, data=df)

    save_model(final_model, f"{output_to_predict}_final_xgboost_model")
    predictions.to_csv(f"{output_to_predict}_predictions.csv")

else:

    predictions = pd.read_csv(f"{output_to_predict}_predictions.csv")
    final_model = load_model(f"{output_to_predict}_final_xgboost_model")

# Plot actual vs. predicted values
plt.figure(figsize=(8,6))
sns.scatterplot(x=predictions[output_to_predict] , y=predictions['prediction_label'])

# plot y = x line
plt.plot([min(predictions[output_to_predict]), max(predictions[output_to_predict])],
         [min(predictions[output_to_predict]), max(predictions[output_to_predict])],
         linestyle = "--", color="red")

plt.xlabel(f"Actual {output_to_predict}")
plt.ylabel(f"Predicted {output_to_predict}")
plt.title(f"Predicted vs. Actual {output_to_predict}")
plt.show()

mystop = 1