import pandas as pd
import numpy as np
import os
import sys
import glob


module_folder = 'Tune_Parameters'
# Import Train_test_split
sys.path.append(module_folder)
from SplitData import data_split_AR

# import Fillna combine
sys.path.append('Data_fillna')
from fillna import FillnaCombine

# import Normalization module
sys.path.append('Data_Normalization_split')
from normalization import Normalization_afterSplit, Denormalize

# import Validation_index
sys.path.append('Validation_index')
from vd_index import rmse, mape, smape, r2, MAE, Each_error_value

# import model
from statsmodels.tsa.arima.model import ARIMA

# impory gridsearch module 
from GridSearch import grid_search_arSeries

# Flag for Find_Economic
find_Economic = False
# Save all Economic_program_name in one list
All_file_name = []
# Create a DataFrame for ours fillna data
# time_index = pd.date_range("2015-01-01", "2021-12-31", freq='D')
# df = pd.DataFrame(index = time_index)
# Coice Economic_prgram

files = glob.glob('Data\\fillna\\*.csv')
for file in files:
    file_name = os.path.basename(file)
    file_name = file_name.split("_")[0]
    All_file_name.append(file_name)
    
# Number each option(Economic_program)
NumberList = [(i, Economic) for i, Economic in enumerate(All_file_name, 1)]
print(NumberList)
Economic_program = input("Please Enter your choice(Enter a number):")

for economic_number in NumberList:
    # (1,Bitcoin)
    # If find the Economial_program
    if economic_number[0] == int(Economic_program):
        # Give it the name of the economic variable
        Economic_program = economic_number[1]
        find_Economic = True
        break
if find_Economic == True:
    pass
# Exit the program
else:
    print("The number you entered is not in the list")
    sys.exit()

# Segment the populated data before standardizing

# Fillna Combine
# df are fillna combine data
df =  FillnaCombine()
# AR model are used for univariate prediction
# So let's take the price of this project as a variable
# The value of the callback is actually Series...
df = df.loc[: , f'{Economic_program}_Price']
# Convert to DataFrame
df = pd.DataFrame(data=df, index=df.index, columns=[f'{Economic_program}_Price'])

print(df.columns)

# Split Train and Test(Use AR-specific methods)
# Because there are only Train and Test data
Train_Data, Test_Data = data_split_AR(df)
# Normalization Dataset(AR models do not need to be standardized)

# Shows how much time the model tooks
# Get the model name using the py file
model_path = os.path.abspath(__file__)
model_name = os.path.basename(model_path)
# Get the model name without .py
model_name = model_name.split(".")[0]

# Parameters adjusted
# The bestmodel here is your model instance that you can use directly to predict
# Find best lag
best_model = grid_search_arSeries(Train_Data, Economic_program, model_name, 5, 5, 5,)

# Use best parameters
p, d, q = best_model.order

# Use ARIMA model
model = ARIMA(Train_Data, order=(p, d, q))
model_fit = model.fit()

# Forecasting Values
y_pred = model_fit.predict(start=len(Train_Data), end=len(Train_Data) + len(Test_Data)-1)

# Calculate the error values
Each_error_value(model_name, Test_Data, y_pred)
