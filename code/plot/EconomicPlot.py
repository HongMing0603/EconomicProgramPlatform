import matplotlib.pyplot as plt
import pandas as pd

# 從CSV文件中讀取數據
bitcoin_prices = pd.read_csv('Data\\fillna\\bitcoin_data_RangeChanges_fillnaData.csv')
brent_prices =pd.read_csv('Data\\fillna\\brent-daily_RangeChanges_fillnaData.csv')
dubai_prices = pd.read_csv('Data\\fillna\\dubai-crude_oil_RangeChanges_fillnaData.csv')
gold_prices = pd.read_csv('Data\\fillna\\Gold-Price_RangeChanges_fillnaData.csv')
wti_prices = pd.read_csv('Data\\fillna\\wti-daily_RangeChanges_fillnaData.csv')


# 提取多個經濟項目的價格數據
bitcoin_prices = bitcoin_prices['Price']
wti_prices = wti_prices["Price"]
brent_prices = brent_prices["Price"]
dubai_prices = dubai_prices["Price"]
gold_prices = gold_prices["Price"]

# 設置圖表樣式
plt.style.use('seaborn')

# 繪製折線圖
plt.plot(bitcoin_prices, label='Bitcoin')
# plt.plot(wti_prices, label='WTI')
# plt.plot(brent_prices, label='Brent-crude')
# plt.plot(dubai_prices, label='Dubai-crude')
plt.plot(gold_prices, label = "gold-Price")

# 添加標題和軸標籤
plt.title('Price Comparison')
plt.xlabel('Date')
plt.ylabel('Price')

# 添加圖例
plt.legend()

# 顯示圖表
plt.show()