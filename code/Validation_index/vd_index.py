import numpy as np
from sklearn.metrics import mean_squared_error
import inspect
import sys

"""
Used to measure error between actual and predicted values
"""
def rmse(y_test, y_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse

def mape(x1, x2):
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)

    return np.mean(abs((x1-x2)/x1))*100


def smape(y_test,y_pred):

    y_test= np.asanyarray(y_test)
    y_pred=np.asanyarray(y_pred)

    return 1/len(y_test) * np.sum(2*np.abs(y_pred-y_test)/(np.abs(y_test)+np.abs(y_pred))*100)

def MAE(y_test, y_pred):
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    mae = np.mean(np.abs(y_test-y_pred))
    return mae

def r2(y_test, y_pred):
    # to array
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    # count mean
    mean_y_true = np.mean(y_test)
    # count sse
    sse = np.sum((y_test-y_pred)**2)
    # count sst
    sst = np.sum((y_test-mean_y_true)**2)
    r2 = 1-(sse/sst)
    return r2

# -->> Lists all custom functions
# Get the current module
current_module = sys.modules[__name__]
# Get the name of each metric
all_function_names = dir(current_module)
# metric_name 
metric_names_obj = {}
for name in all_function_names:
    # getattr: return the name of object
    # obj is metric obj(There may be native parameters)
    obj = getattr(current_module, name)
    if callable(obj) and inspect.getmodule(obj) == current_module and not name.startswith("__"):
        metric_names_obj[name] = obj
        

def Each_error_value(model_name,y_Test, y_Pred):
    """
    Allow y_Test and y_Pred to be imported into
    the evaluation metrics themselves
    Arg:
        model: The model taht was originally called
        y_Test:y_Test DataFrame
        y_Pred:y_Pred DataFrame
    output:Output the values of the various evaluation metrices
    """
    print(metric_names_obj)
    print(f"Model:{model_name}")
    # Iterate through each metric
    for metric_name in metric_names_obj:
        metric_obj = metric_names_obj[metric_name]
        index_error = metric_obj(y_Test, y_Pred)
        print(f"{metric_name}_value: {index_error}")
