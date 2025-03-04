import pandas as pd
import os
import pycaret
from pycaret.regression import *
import matplotlib.pyplot as plt
import seaborn as sns

output_to_predict = "Flanker_Standard_Age_Corrected"

working_dir = os.getcwd()

df = pd.read_csv(f"{working_dir}/final_df_for_xgboost.csv")

# Remove rows where the response variable is NA
df = df.dropna(subset=[output_to_predict])

#Encode Sex column Female = 0 Male = 1
df = df["Sex"].replace({"Female": 0, "Male": 1})

# Set up 10-fold cross validation
exp = setup(
    data=df,
    target=output_to_predict,
    train_size=0.8,
    session_id=123,
    fold=10
)

# Tune XGBoost model with Bayesion optimization
xgb = create_model("xgboost")

tuned_xgb = tune_model(
    xgb,
    optimize="R2",
    fold=10,
    search_algorithm="bayesian",
    n_iter=50
)

# Finalize model and predict
final_xgb = finalize_model(tuned_xgb)
predictions = predict_model(final_xgb, data=df)

# Plot actual vs. predicted values
plt.figure(figsize=(8,6))
sns.scatterplot(x=predictions[output_to_predict])

# plot y = x line
plt.plot([min(predictions[output_to_predict]), max(predictions[output_to_predict])],
         min(predictions[output_to_predict]), max(predictions[output_to_predict]),
         linestyle = "--", color="red")

plt.xlabel("Actual Y")
plt.ylabel("Predicted Y")
plt.title("Predicted vs. Actual Y")
plt.show()

mystop = 1