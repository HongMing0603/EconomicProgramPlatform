import pandas as pd
import numpy as np
import os
import sys
import glob

from sklearn.linear_model import BayesianRidge

# Import GridSearch_module
module_folder = 'Tune_Parameters'
sys.path.insert(0, module_folder)
import GridSearch
from GridSearch import run_grid_search
from BayesSearch import BayesSearch
# Import Train_test_split
sys.path.append(module_folder)
from SplitData import data_split

# import Fillna combine
sys.path.append('Data_fillna')
from fillna import FillnaCombine

# import Normalization module
sys.path.append('Data_Normalization_split')
from normalization import Normalization_afterSplit, Denormalize

# import Validation_index
sys.path.append('Validation_index')
from vd_index import Each_error_value

# import model
from sklearn.linear_model import ElasticNet

def elasticNet(X, y, config = None):


    # Split Train and Test
    X_train, X_test, y_train, y_test = data_split(X,y)

    # Normalization Dataset
    X_train_scalered = Normalization_afterSplit("X_scaler", X_train, "train")
    X_test_scalered = Normalization_afterSplit("X_scaler", X_test, "test")
    y_train_scalered = Normalization_afterSplit("y_scaler", y_train, "train")

    # Get the model name using the py file
    model_path = os.path.abspath(__file__)
    model_name = os.path.basename(model_path)
    # Get the model name without .py
    model_name = model_name.split(".")[0]

    # Create a model
    model = ElasticNet()
    param_grid = {
        "l1_ratio":[0.1, 0.5, 0.9]
    }

    # store Predicted variable
    Predicted_program = config["Y_name"]

    # Shows how much time the model tooks
    # Get the model name using the py file
    model_path = os.path.abspath(__file__)
    model_name = os.path.basename(model_path)
    # Get the model name without .py
    model_name = model_name.split(".")[0]


    # Parameters adjusted
    # The bestmodel here is your model instance that you can use directly to predict
    # Pass Economic_name for GridSearch (Economic_program)
    best_model_gridSearch = run_grid_search(model, param_grid, X_train_scalered, y_train_scalered, config, model_name)
    # Use BayesSearch
    best_model_BayesSearch = BayesSearch(model, param_grid, X_train_scalered, y_train_scalered, config, model_name)

    # Forecasting Values
    y_pred_GridSearch = best_model_gridSearch.predict(X_test_scalered) 
    y_pred_BayesSearch = best_model_BayesSearch.predict(X_test_scalered) 

    # Get ours predict economi project name
    economi_project_name = Predicted_program
    economi_project_name = economi_project_name.split('_')[0]

    # Create a dataFrame for y_pred 
    y_pred_GridSearch = pd.DataFrame(data=y_pred_GridSearch, index=y_test.index, columns=["y_Pred"])
    y_pred_BayesSearch = pd.DataFrame(data=y_pred_BayesSearch, index=y_test.index,columns=["y_Pred"])
    # Denormalize it (y_pred)
    y_pred_GridSearch = Denormalize(y_pred_GridSearch)
    y_pred_BayesSearch = Denormalize(y_pred_BayesSearch)

    y_pred_GridSearch = pd.DataFrame(data=y_pred_GridSearch, columns=["y_Pred"])
    y_pred_BayesSearch = pd.DataFrame(data=y_pred_BayesSearch, columns=["y_Pred"])

    y_pred_GridSearch.index = y_test.index
    y_pred_BayesSearch.index = y_test.index

    # For Grid Search dataFrame combine
    print(y_pred_GridSearch.columns)
    combine_df_GridSearch = pd.concat([y_test, y_pred_GridSearch], axis=1)
    combine_df_GridSearch.columns.values[0] = "y_Test"
    combine_df_GridSearch.columns.values[1] = "y_Pred"

    # For BayesSearch DataFrame combine
    print(y_pred_BayesSearch.columns)
    combine_df_BayesSearch = pd.concat([y_test, y_pred_BayesSearch], axis=1)
    combine_df_BayesSearch.columns.values[0] = "y_Test"
    combine_df_BayesSearch.columns.values[1] = "y_Pred"


    # Calculate the error values
    # For Grid Search
    print("Grid Search Validation_index:")
    Each_error_value(model_name, combine_df_GridSearch["y_Test"], combine_df_GridSearch["y_Pred"])

    # For BayesSearch
    print("Bayes Search Validation_index:")
    Each_error_value(model_name, combine_df_BayesSearch["y_Test"], combine_df_BayesSearch["y_Pred"])
