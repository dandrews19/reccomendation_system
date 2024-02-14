from xgboost import XGBRegressor
import optuna
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle
from prep_data_for_model import prep_data_for_model
import os
import sys
from pyspark import SparkContext

# Configure PySpark to use a specific Python interpreter based on the command line argument
os.environ['PYSPARK_PYTHON'] = sys.argv[2]

# Initialize a Spark context for distributed data processing
sc = SparkContext('local[*]', 'recommend')

# Load training and validation datasets along with their attributes
train_df, validation_df, attributes = prep_data_for_model(sc, 'data')


def objective(trial):
    """
    Defines the objective function for Optuna's hyperparameter optimization.

    Parameters:
    - trial: An Optuna trial object used to suggest values for the model's hyperparameters.

    Returns:
    - rmse: The root mean squared error of the model predictions on the validation set.
    """
    # Suggest hyperparameter values for the XGBRegressor model
    param = {
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0),  # L2 regularization term
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0),  # L1 regularization term
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
        # Subsample ratio of columns when constructing each tree
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),  # Subsample ratio of the training instances
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        # Step size shrinkage used in update to prevent overfitting
        'max_depth': trial.suggest_int('max_depth', 3, 10),  # Maximum depth of a tree
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
        # Minimum sum of instance weight (hessian) needed in a child
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),  # Number of gradient boosted trees
        'n_jobs': -1  # Use all available cores for training
    }

    # Initialize XGBRegressor with suggested parameters
    model = XGBRegressor(**param)

    # Prepare training features and target variable
    X_train = train_df[attributes]
    y_train = train_df['true_rating']

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Predict on the validation set
    preds = model.predict(validation_df[attributes])

    # Calculate RMSE on the validation set and return it as the objective value
    rmse = np.sqrt(mean_squared_error(validation_df['true_rating'], preds))
    return rmse


# Initialize Optuna study for hyperparameter optimization, minimizing RMSE
study = optuna.create_study(direction='minimize')
# Run optimization with 100 trials
study.optimize(objective, n_trials=100)

# Output the results of the best trial
trial = study.best_trial
print(f'Best trial: {trial.values}')
print(f'Best parameters: {trial.params}')

# Train final model with best parameters found
final_model = XGBRegressor(**trial.params)
X_train = train_df[attributes]
y_train = train_df['true_rating']
final_model.fit(X_train, y_train)

# Save the final model to a pickle file for later use
with open('final_xgb_model.pickle', 'wb') as file:
    pickle.dump(final_model, file)
