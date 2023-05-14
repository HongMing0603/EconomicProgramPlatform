import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import numpy as np
import os

def standardScaler(data_name):
    data = pd.read_csv(f'data/{data_name}.csv')
    # 需要處理資料
    # 設定Mean值
    data["Mean"] = (data["High"]+data["Low"])/2
    data = data.set_index("Date")
    data.index = pd.to_datetime(data.index)
    # Using Low High Open Predict Mean
    X = data[['High', 'Low', 'Open']]
    y = data[['Mean']]

    # Fillna
    imputer = SimpleImputer(strategy='mean')
    X_data_filled = imputer.fit_transform(X)
    y_data_filled = imputer.fit_transform(y)
    # counting the Short Term 、LongTerm days
    data_len = len(X_data_filled)
    # 短期預測:1年以下 中期預測1~5年 長期預測5年以上
    # 判斷是否可以分成長期資料
    if data_len >= 365*5:
        Long_term_test_size = (365*5)/data_len
        Mid_term_test_size = (365*3)/data_len
        Short_term_test_size = (365*1)/data_len
    elif data_len >= 365*3:
        Long_term_test_size = False
        Mid_term_test_size = (365*3)/data_len
        Short_term_test_size = (365*1)/data_len
    elif data_len >=365:
        Long_term_test_size = False
        Mid_term_test_size = False
        Short_term_test_size = (365*1)/data_len

    # Choice 短中長期預測
    predict_choice = input('請輸入數字1=短期預測  2=中期預測  3=長期預測')
    choice_dict = {"1":Short_term_test_size,
                    "2":Mid_term_test_size,
                    "3":Long_term_test_size}
    # 假如輸入的數字有在字典當中，則給予相對應選擇的變量
    if predict_choice in choice_dict:
        predict_selection = choice_dict[predict_choice]
    # data split
    X_train, X_test, y_train, y_test = train_test_split(X_data_filled, y_data_filled, shuffle=False, test_size=predict_selection)
    # Normalization
    scaler = StandardScaler()
    # X_train, y_train, X_test need Normalization
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    y_train = scaler.fit_transform(y_train)

    # Create DataFrame
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)
    y_train_df = pd.DataFrame(y_train)
    y_test_df = pd.DataFrame(y_test)

    output_df = pd.concat([X_train_df, X_test_df, y_train_df, y_test_df], axis=1)
    output_df.columns = ['X_train_High','X_train_Low','X_train_Open', 'X_test_High','X_test_Low','X_test_Open', 'y_train', 'y_test']
    print(output_df)
    
    # 假如沒有data資料夾則創建data資料夾
    if os.path.exists('data/standardScaler'):
        print('已有資料夾~下一步')
    else:
        os.makedirs('data/standardScaler')
    # 把資料用csv格式輸出
    output_df.to_csv(f'data/standardScaler/{data_name}_standardScalerData.csv', index=False, mode='w')
    print("寫入成功!!")
if __name__ == "__main__":
    standardScaler("bitcoin_data")
