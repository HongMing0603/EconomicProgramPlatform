
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import joblib
import os
import time
import numpy as np
np.int = int

model_folder = "Data/BayesSearchModel/"

def BayesSearch(model, params,X_train, y_train, config_name, model_name):
    """
    config_name: Some command variable include What are the variable in X, What are the variable of Y...
    """

    X_name = config_name["X_name"]
    Y_name = config_name['Y_name']
    Bayes_search_model_name = f"{model_name}_BayesSearch_{X_name}2{Y_name}"
    # Determine whethere the parameter has been searched
    # and saved
    best_model = Determine_model(model_folder, Bayes_search_model_name)
    # best_model Already exists
    if best_model != "":
        # Show how search cost
        elapsed_parameters(model_name, best_model)
        return best_model
    else:

        opt = BayesSearchCV(model, params, cv=5)
        # Calculate the time
        start = time.time()
       
        
        opt.fit(X_train, y_train)
        end = time.time()
        elapsed_time = end-start
        # Add time to object
        opt.elapsed_time = elapsed_time
        # Output the result
        elapsed_parameters(model_name, opt)
        # save model
        joblib.dump(opt, model_folder + Bayes_search_model_name)
        # return best model
        return opt

def Determine_model(model_folder, Bayes_search_model_name):
    """
    Determine whether model exists 
    """
    # best_model_file
    best_model_file = model_folder + Bayes_search_model_name
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
