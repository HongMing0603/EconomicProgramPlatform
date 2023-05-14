
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import warnings
warnings.filterwarnings("ignore")

# 導入分割好的資料
data = pd.read_csv("data/standardScaler/bitcoin_data_standardScaler.csv")
print(data)

# 重新引用X_train X_test...



# Load BayesianRegression from sklearn
reg= linear_model.BayesianRidge()

# Gird search cv
from sklearn.model_selection import GridSearchCV
bs_parameters={
    "alpha_1":[1e-5, 1e-6, 1e-7],
    "alpha_2":[1e-5, 1e-6, 1e-7],
    "lambda_1":[1e-5, 1e-6, 1e-7],
    "lambda_2":[1e-5, 1e-6, 1e-7],
}
bs_gs = GridSearchCV(estimator= reg, param_grid=bs_parameters, n_jobs=-1, cv=5)
start = time.time()
bs_gs.fit(Xtrain.reshape((len(Xtrain),1)), ytrain)
end = time.time()
print(bs_gs.best_estimator_)
print("Time cost:",end-start)
# BayesianRidge(alpha_1=1e-05, alpha_2=1e-05, lambda_1=1e-07, lambda_2=1e-05)

# BaysianSearch + cv
from skopt import BayesSearchCV 
from skopt.space import Real, Categorical, Integer
bs_parameters={
    "alpha_1":[1e-5, 1e-6, 1e-7],
    "alpha_2":[1e-5, 1e-6, 1e-7],
    "lambda_1":[1e-5, 1e-6, 1e-7],
    "lambda_2":[1e-5, 1e-6, 1e-7],
}

reg = linear_model.BayesianRidge()
bayesgs = BayesSearchCV(estimator= reg, search_spaces=bs_parameters,n_iter=20, n_jobs=-1, verbose=2, cv= 3)
start = time.time()
bayesgs.fit(Xtrain.reshape((len(Xtrain),1)), ytrain)
end = time.time()
print(bayesgs.best_estimator_)
print("Time cost:",end-start)
# BayesianRidge(alpha_1=1e-05, lambda_1=1e-07, lambda_2=1e-05)


reg = linear_model.BayesianRidge(alpha_1=1e-05, lambda_1=1e-07, lambda_2=1e-05)
reg.fit(Xtrain.reshape((len(Xtrain),1)), ytrain)
ypred=reg.predict(Xtest.reshape((len(Xtest),1)))
ytest=ytest.reshape((674,1))

#plot the same
plt.plot(arr,label='actual')
plt.plot(ypred,label='predicted')
plt.legend()
plt.show()

#Report the RMSE
c=0
for i in range(674):
    c+=(ypred[i]-ytest[i])**2
c/=674
print("RMSE:",c**0.5 +201)

# MAPE
c=0
for i in range(674):
    c+=abs((ypred[i]-ytest[i])/ytest[i])
c/=674
print("MAPE:",c*100)

# SMAPE
c=0
for i in range(674):
    c+= abs(ypred[i]-ytest[i])/((ytest[i]+ypred[i])/2)
c/=674
print("SMAPE:",c*100)

print("BAYESIAN REGRESSION")
print("Mean value depending on open")

# 保存模型
model_directory = "model"
model_file_name = f'{model_directory}/bayesian.pkl'
joblib.dump(reg, model_file_name)