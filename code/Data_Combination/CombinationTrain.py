import os
import glob
import pandas as pd
import sys

# 導入各個模型
# import BayesRidgeRegression
sys.path.append('Train_Prediction')
from BayesianRidgeRegression import bayesRidgeRegression
# Import ElasticNet Model
from ElasticNet import  elasticNet
# Import Polynomial Regression
from PolynomialRegression import polynomialRegression
# Import RandomForest Regression
from RandomForestRegression import randomForest
# Import SupportVectorRegression
from SupportVectorRegression import supportVectorRegression


# 獲取組合資料

# 文件夹路径
folder_path = 'Data\\Combination'

# 获取文件夹下所有的CSV文件
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# 預測項目csv位置
# filepath = 'Data\\fillna\\bitcoin-data_RangeChanges_fillnaData.csv'

# 所有y的可能 (名字來自於fillna的combine)
economic_list = ['bitcoin-data_Price', 'brent-daily_Price', 'dubai-daily_Price', 'Gold-Price_Price', 'wti-daily_Price']
# 把一個經濟項目當作y

# csv的位置
file = 'Data/fillna/Combine/fillna_Combine.csv'

# 讀取csv
file_df = pd.read_csv(file, index_col='Date')

# 讀取比特幣資料 當作y
# 遍歷一遍經濟項目變量當作y
for economic_program in economic_list:
  
    # get economic program name
    economic_name = economic_program.split("_")[0]

    # 設定y變量
    y_df = file_df[[economic_program]].copy()
    # 設定x變量
    x_df = file_df.drop(economic_program, axis=1)
    # try to drop the high low column(If exist above column,The mean loss of predict)
    drop_column = [f"{economic_name}_High", f"{economic_name}_Low"]
    for column in drop_column:
        # try to delete column(if it's exist)
        try:
            # if you already know the economic's High and low, you lose the meaning of predict
            x_df = x_df.drop(column, axis=1)
        except:
            pass
    # Transfer to predict
    # 創建一個字典用於儲存配置信息(導出模型或csv是根據這個名稱來製作的)
    # Config包含(xname與y_name)
    config = {}
    config["X_name"] = "All"
    config["Y_name"] = f"{economic_program}"
    bayesRidgeRegression(x_df, y_df, config)
    elasticNet(x_df, y_df, config)
    polynomialRegression(x_df, y_df, config)
    randomForest(x_df, y_df, config)
    supportVectorRegression(x_df, y_df, config)
    


# # 存放被預測變數的名稱
# y_name = filepath.split("\\")[2]
# y_name = y_name.split("_")[0]
# config["Y_name"] = y_name




# # 存储CSV文件的内容的列表
# csv_data = []

# # 讀取以下內容當作X
# # 逐个读取CSV文件并将内容存储到列表中
# for csv_file in csv_files:
#     X = pd.read_csv(csv_file, index_col="Date")
#     # 顯示現在用甚麼當作X?
#     X_variable = os.path.basename(csv_file)
#     X_variable = X_variable.split(".")[0]
#     config['X_name'] = X_variable
#     print(f"現在使用的X為: {X_variable}")

#     # 傳送資料到各模型進行訓練?
#     bayesRidgeRegression(X, y, config)
#     elasticNet(X, y, config)
#     polynomialRegression(X, y, config)
#     randomForest(X, y, config)
#     supportVectorRegression(X, y, config)







