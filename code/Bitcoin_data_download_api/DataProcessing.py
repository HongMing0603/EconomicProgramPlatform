# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 15:06:36 2023

@author: user
"""
import requests
import pandas as pd
import datetime
import time
import csv
import os
import sys
#讓日期可以取到現在
date = datetime.datetime.now()
#轉換成時間戳記
timeStamp = int(date.timestamp())
print(timeStamp)
# It is used to determine whether the number entered by the
# user corresponds to the economic item
find_Economic = False

# 可供選擇的加密貨幣
coin_list = ["bitcoin", "ethereum"]
coin_list = [(num, coin ) for num, coin in enumerate(coin_list, 1)]

while True:
    print(coin_list)
    coin = input("please enter the currency you want to select(Enter number): ")

    for economic_number in coin_list:
        # (1,Bitcoin)
        # If find the Economial_program
        if economic_number[0] == int(coin):
            # Give it the name of the economic variable
            Economic_program = economic_number[1]
            find_Economic = True
            break
    if find_Economic == True:
        break
    # Exit the program
    else:
        print("The number you entered is not in the list")
        
coin = Economic_program
    


# Fetching data from the server
url = "https://web-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"
# param = {"convert":"USD","slug":"bitcoin","time_end":"1601510400","time_start":"1367107200"}
param = {"convert":"USD","slug":coin,"time_end":timeStamp,"time_start":"1367107200"}

content = requests.get(url=url, params=param).json()
df = pd.json_normalize(content['data']['quotes'])


# import os
# print(os.getcwd())
# oil = pd.read_csv("oil\wti-daily.csv",encoding='utf-8')

# Extracting and renaming the important variables
df['Date']=pd.to_datetime(df['quote.USD.timestamp']).dt.tz_localize(None)


df['Low'] = df['quote.USD.low']
df['High'] = df['quote.USD.high']
df['Open'] = df['quote.USD.open']
# 把名字轉換後，刪除原本數據

# Drop original and redundant columns
df=df.drop(columns=['time_open','time_close','time_high','time_low', 'quote.USD.low', 'quote.USD.high', 'quote.USD.open', 'quote.USD.close', 'quote.USD.volume', 'quote.USD.market_cap', 'quote.USD.timestamp'])
# 換了名稱並且捨去原本名稱
# 捨去不要的數據

# Creating a new feature for better representing day-wise values
df['Price'] = (df['Low'] + df['High'])/2

# Cleaning the data for any NaN or Null fields
# 除去空值
df = df.dropna()

# Creating a copy for making small changes
dataset_for_prediction = df.copy()



dataset_for_prediction=dataset_for_prediction.dropna()
# 去除預測值的空值

# date time typecast
dataset_for_prediction = dataset_for_prediction.set_index('Date')
dataset_for_prediction.index = pd.to_datetime(dataset_for_prediction.index, unit='D')
# Only the part of the date is kept
dataset_for_prediction.index = dataset_for_prediction.index.date
# 把Date設為Time Series index
print(dataset_for_prediction)
# Reorder index
reorder_index = ['Price', 'Open', 'High', 'Low']
dataset_for_prediction = dataset_for_prediction.reindex(columns=reorder_index)

# 假如沒有data資料夾則創建data資料夾
if os.path.exists('data'):
    print('已有資料夾~下一步')
else:
    os.makedirs('data')
# 把資料用csv格式輸出
dataset_for_prediction.to_csv(f'Data/crypto_downloader/{coin}_data.csv', index=True, mode='w', index_label="Date")
print("寫入成功!!")