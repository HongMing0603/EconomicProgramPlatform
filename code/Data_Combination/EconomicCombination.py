import itertools
import glob
import os
import pandas as pd


# 获取所有 CSV 文件的文件名
files = glob.glob('Data\\fillna\\*.csv')

# 定义用于保存文件名的列表
All_file_name = []

# 遍历每个 CSV 文件，提取文件名并保存到列表
for file in files:
    file_name = os.path.basename(file)
    file_name = file_name.split("_")[0]
    All_file_name.append(file_name)


# 定义比特币、石油和黄金的列表
bitcoin = ['bitcoin-data']
oil = ['brent-daily', 'dubai-daily', 'wti-daily']
gold = ['Gold-Price']

# 定義誰為y
y = bitcoin
y_name = y[0]

# 定義多個組合(如果預測的是bitcoin X裡面還有bitcoin這樣有意義?有意義的話是要使用滑窗嗎?)
combinations = [
    
    [oil,"oil"],
    [gold,"gold"],
    [gold + oil,"gold--oil"],
    [gold + oil + bitcoin, "gold--oil--bitcoin"]
    
    # 在這裡添加更多的組合
]

# 创建一个空的 DataFrame 用于存储合并后的数据
merged_data = pd.DataFrame()
# 把這幾個分別轉成csv文件?
for X_project in combinations:
    X = X_project[0]
    # 我把名稱放在最後一個位置
    X_name = X_project[-1]

# 提取 X 对应的 CSV 数据
    X_data = pd.DataFrame()
    for file_name in X:
        if file_name in All_file_name:
            file_path = os.path.join("Data/fillna/", file_name + "_RangeChanges_fillnaData.csv")  # 替换成你的文件路径
            data = pd.read_csv(file_path)
        # 使用try-except语句处理可能不存在的列
        try:
            # 使用文件名作为列名
            column_name = file_name
            
            # 尝试读取'Open'、'High'和'Low'列
            price_data = data['Price']
            X_data[column_name + '_Price'] = price_data  

            open_data = data['Open']
            X_data[column_name + '_Open'] = open_data

            high_data = data['High']
            X_data[column_name + '_High'] = high_data
            
            low_data = data['Low']
            X_data[column_name + '_Low'] = low_data
                      
            
        except KeyError:
            # 如果某列不存在，则忽略并继续处理下一个文件
            continue


    # 去除y的項目只保留x項目
    try:
        drop_name = f"{y_name}_Price"
        X_data =  X_data.drop(drop_name, axis=1)
    except:
        pass
    # 把組合的結果轉成csv
    # 設定index為原本csv的日期
    X_data.index = data["Date"]
    X_data.to_csv(f'Data/Combination/{X_name}.csv', index= True)








