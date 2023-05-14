from sklearn.model_selection import GridSearchCV
import time
import joblib
import sys
import os
import pmdarima
from pmdarima.arima import auto_arima


def run_grid_search(model, param_grid, X_train, y_train, config_name, model_name):
    """
    Arg:
        model:Your model
        param_grid:You select the range to adjust for the model's parameter
        X_train
        y_train
        config_name: Some command variable include What are the variable in X, What are the variable of Y...
        model_name: if you use BayesianRidgeRegression you just type his name
    return:
        grid_search.best_estimator:Instantiation of optimal model parameters
        time_cost:How much time it took to find the parameters
    """

    X_name = config_name["X_name"]
    Y_name = config_name['Y_name']
    model_folder = "Data/GridSearchModel/"
    grid_search_model_name = f"{model_name}_GridSearch_{X_name}2{Y_name}"
    # Determine whether the model exists
    best_model = Determine_model(model_folder, grid_search_model_name)
    if best_model != "":
        # Already GridSearch
        # Display the search time
        
        # Shows how much time the parameter search
        # took and show best parameters
        elapsed_parameters(model_name, best_model)

        return best_model
    else:
        grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, error_score='raise')
        start = time.time()
        grid_search.fit(X_train,y_train)
        end = time.time()
        # Calculate the time it takes
        elapsed_time = end-start
        # best model 
        best_model = grid_search
        # Write the time record into it as well
        best_model.elapsed_time = elapsed_time
        # Shows how much time the parameter search
        # took and show best parameters
        elapsed_parameters(model_name, best_model)

        # Save Best Parameters model
        
        # Judgement the folder exist or not
        # Assume the folder does not exist
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        else:
            print("The folder already exists, come on!!")


        joblib.dump(best_model, model_folder + grid_search_model_name)
        return best_model

def grid_search_allin(model, param_grid, X_train, y_train, model_name):
    pass   

def Determine_model(model_folder, grid_search_model_name):
    """
    Determine whether model exists 
    """
    # best_model_file
    best_model_file = model_folder + grid_search_model_name
    # Determine whether best_model_exists
    if os.path.exists(best_model_file):
        # exists
        # Directory return file without grid_search(Because GridSearch compleate)
        best_model = joblib.load(best_model_file)
        return best_model
    else:
        return ""

def elapsed_parameters(model_name, best_model):
    """
    Displays the parameter search time and the best parameters
    """
    print(f"{model_name}_Model parameter tunning took these times:{best_model.elapsed_time} seconds")
    # Display the best model and parameters
    best_parameters = best_model.best_params_
    print(f"{model_name}'s best parameters are: {best_parameters}")

def elapsed_parameters_arimaSeries(model_name, best_model):
    """
    Displays the parameter search time and the best parameters
    """
    print(f"{model_name}_Model parameter tunning took these times:{best_model.elapsed_time} seconds")
    # Display the best model and parameters
    best_parameters = best_model.order
    print(f"{model_name}'s best parameters are: {best_parameters}")

def grid_search_arSeries(train_Data, Economic_program, model_name,p=0 , d=0, q=0,):
    """
    Arg:
        p: P The maximum search scope
        d: d The maximum search scope
        q: q The maximum search scope
    """
    model_folder = "Tune_Parameters/GridSearchModel/"
    grid_search_model_name = f"{model_name}_GridSearch_{Economic_program}"
    # Determine whether the model exists
    best_model = Determine_model(model_folder, grid_search_model_name)
    if best_model != "":
        # Already GridSearch
        # Display the search time
        
        # Shows how much time the parameter search
        # took and show best parameters
        elapsed_parameters_arimaSeries(model_name, best_model)

        return best_model
    else:
        # Determine if it's ARIMA
        # not d=0, yes d=d(user setting)
        if model_name != "ARIMA":
            start = time.time()
            best_order = auto_arima(train_Data, start_p=0, start_q=0,
                        # The user selects the search scope
                    max_P=p, max_q=q, d=0,
                        seasonal=False,
                    trace=True, error_action="ignore",
                    suppress_warnings=True, stepwise=True)
            end = time.time()
        else: # is ARIMA
            start = time.time()
            best_order = auto_arima(train_Data, start_p=0, start_q=0,
                        # The user selects the search scope
                    max_P=p, max_q=q, max_d=d,
                        seasonal=False,
                    trace=True, error_action="ignore",
                    suppress_warnings=True, stepwise=True)
            end = time.time()
        # Calculate the time it takes
        elapsed_time = end-start
 
        # Write the time record into it as well
        best_order.elapsed_time = elapsed_time
        best_model = best_order
        # Shows how much time the parameter search
        elapsed_parameters_arimaSeries(model_name, best_model)
        # Save Best Parameters model
        
        # Judgement the folder exist or not
        # Assume the folder does not exist
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        else:
            print("The folder already exists, come on!!")


        joblib.dump(best_order, model_folder + grid_search_model_name)
        return best_order

    