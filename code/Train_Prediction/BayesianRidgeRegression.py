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


def bayesRidgeRegression(X, y, config = None):
    """
    X: 用甚麼來預測
    y: 你想預測甚麼
    config: 一些會使用到的名稱 像是X使用甚麼經濟項目
    """

    X = X
    y = y

    # 預測經濟項目名稱拿得到?
    # 被預測項目名稱
    Predicted_program = config["Y_name"]

    # X project name used
    X_project = config["X_name"]
    

    # Split Train and Test
    X_train, X_test, y_train, y_test = data_split(X,y)

    # Normalization Dataset
    X_train_scalered = Normalization_afterSplit("X_scaler", X_train, "train")
    X_test_scalered = Normalization_afterSplit("X_scaler", X_test, "test")
    y_train_scalered = Normalization_afterSplit("y_scaler", y_train, "train")

    # Convert the scaled data into dataframes
    X_train_df = pd.DataFrame(X_train_scalered, columns=X_train.columns, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_scalered, columns=X_test.columns, index=X_test.index)
    y_train_df = pd.DataFrame(y_train_scalered, columns=y_train.columns, index=y_train.index)
    y_test_df = pd.DataFrame(y_test, columns=y_test.columns, index=y_test.index)

    # Concatenate the dataframes
    train_df = pd.concat([X_train_df, y_train_df], axis=1)

    # Write the dataframes to CSV files
    train_df.to_csv(f"Data\\Normalization\\split_to_XY\\{X_project}2{Predicted_program}-train.csv", index=True)
    X_test_df.to_csv(f"Data\\Normalization\\split_to_XY\\{X_project}2{Predicted_program}-test.csv", index=True)
    y_test_df.to_csv(f"Data\\Normalization\\split_to_XY\\{X_project}2{Predicted_program}-y_test.csv", index=True)
    # Shows how much time the model tooks
    # Get the model name using the py file
    model_path = os.path.abspath(__file__)
    model_name = os.path.basename(model_path)
    # Get the model name without .py
    model_name = model_name.split(".")[0]

    # Create a model
    model = BayesianRidge()
    param_grid = {
        "alpha_1":[1e-5, 1e-6, 1e-7],
        "alpha_2":[1e-5, 1e-6, 1e-7],
        "lambda_1":[1e-5, 1e-6, 1e-7],
        "lambda_2":[1e-5, 1e-6, 1e-7],
    }
    # Parameters adjusted

    # Set the y_pred index from y_test.index

    # Parameters adjusted
    # The bestmodel here is your model instance that you can use directly to predict
    # Pass Economic_name for GridSearch (Economic_program)
    best_model_gridSearch = run_grid_search(model, param_grid, X_train_scalered, y_train_scalered, config, model_name)
    # Use BayesSearch
    best_model_BayesSearch = BayesSearch(model, param_grid, X_train_scalered, y_train_scalered, config, model_name)

    # Forecasting Values
    y_pred_GridSearch = best_model_gridSearch.predict(X_test_scalered) 
    y_pred_BayesSearch = best_model_BayesSearch.predict(X_test_scalered) 

    

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
    # print(y_pred_BayesSearch.columns)
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

def BRR_allIn(Xdf, ydf, config):
    """
    Use total column to predict specific column
    Xdf:x data frame
    ydf:y data frame
    config:Include X name and y name
    """
    # split data set
    X = Xdf
    y = ydf
    # Split Train and Test
    X_train, X_test, y_train, y_test = data_split(X,y)

    # Normalization
    # Normalization Dataset
    X_train_scalered = Normalization_afterSplit("X_scaler", X_train, "train")
    X_test_scalered = Normalization_afterSplit("X_scaler", X_test, "test")
    y_train_scalered = Normalization_afterSplit("y_scaler", y_train, "train")

    # Convert the scaled data into dataframes
    X_train_df = pd.DataFrame(X_train_scalered, columns=X_train.columns, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_scalered, columns=X_test.columns, index=X_test.index)
    y_train_df = pd.DataFrame(y_train_scalered, columns=y_train.columns, index=y_train.index)
    y_test_df = pd.DataFrame(y_test, columns=y_test.columns, index=y_test.index)

    # Get model name
    # Get the model name using the py file
    model_path = os.path.abspath(__file__)
    model_name = os.path.basename(model_path)
    # Get the model name without .py
    model_name = model_name.split(".")[0]

    # Create a model
    model = BayesianRidge()
    param_grid = {
        "alpha_1":[1e-5, 1e-6, 1e-7],
        "alpha_2":[1e-5, 1e-6, 1e-7],
        "lambda_1":[1e-5, 1e-6, 1e-7],
        "lambda_2":[1e-5, 1e-6, 1e-7],
    }

    # Parameters adjusted
    # The bestmodel here is your model instance that you can use directly to predict
    # Pass Economic_name for GridSearch (Economic_program)
    best_model_gridSearch = run_grid_search(model, param_grid, X_train_scalered, y_train_scalered, config, model_name)
    # Use BayesSearch
    best_model_BayesSearch = BayesSearch(model, param_grid, X_train_scalered, y_train_scalered, config, model_name)

    # Forecasting Values
    y_pred_GridSearch = best_model_gridSearch.predict(X_test_scalered) 
    y_pred_BayesSearch = best_model_BayesSearch.predict(X_test_scalered) 

    

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
    # print(y_pred_BayesSearch.columns)
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
