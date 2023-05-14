import pandas as pd
import requests

# 設定 API 金鑰和網址
api_key = '6J28Q8XNBY7OOJ0U'  # 替換成您的 Alpha Vantage API 金鑰
url = 'https://www.alphavantage.co/query'

# 設定查詢參數
params = {
    'function': 'TIME_SERIES_DAILY',
    'symbol': 'XAU',  # XAU 是黃金代號
    'outputsize': 'full',
    'apikey': api_key
}

# 發送 API 請求並取得回應
response = requests.get(url, params=params)
data = response.json()

# 將資料轉換成 DataFrame
df = pd.DataFrame(data['Time Series (Daily)']).T

# 轉換索引為日期型態
df.index = pd.to_datetime(df.index)

# 篩選指定日期範圍
start_date = '2015-01-01'
end_date = pd.Timestamp.now().date().strftime('%Y-%m-%d')
df = df.loc[start_date:end_date]
