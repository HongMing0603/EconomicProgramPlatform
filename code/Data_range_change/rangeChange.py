import pandas as pd
import glob
import os
import datetime
# Find the files
files = glob.glob('economi_project/*.csv')
print(files)

# Set the columns that I want drop
drop_columns = ['Vol.', 'Change %', 'Volume', 'Chg%']

for file in files:
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df = df.sort_index(ascending=True)

    delete_list_for_each_csv = []
    # Determine if there are columns to delete
    for i in df.columns:
        if i in drop_columns:
            delete_list_for_each_csv.append(i)
            print(delete_list_for_each_csv)

    # 取最新的日期
    # 實驗用暫訂 2023-01-23
    date = "2023-01-23"
    # date = datetime.date.today().strftime('%Y-%m-%d')
    # Change Datetime
    df = df.loc["2015-01-01":date]

    # Delete unecessary column
    if len(delete_list_for_each_csv) >0:
        df = df.drop(columns=delete_list_for_each_csv)
        print(f"Delete column:{delete_list_for_each_csv}")
    # get file_name
    file_name = os.path.basename(file)
    # Delete .csv
    file_name = file_name.replace(".csv", "")
    print(file_name)
 
    # Create a Directory
    if not os.path.exists("Data/RangeChanges"):
        os.makedirs("Data/RangeChanges")
    else:
        print("Directory already exists")

    # Create a Empty DataFrame
    date_rng = pd.date_range(start='2015-01-01', end= date, freq='D')
    df_empty = pd.DataFrame(index=date_rng)
    print(df_empty)
    
    # Merge two DataFrame
    df = pd.merge(df_empty, df, how='left', left_index=True, right_index=True)
    # Just save column in [Price, Open, High, Low] 
    
    # Output to csv
    df.to_csv(f"Data/RangeChanges/{file_name}_RangeChanges.csv", index=True)

    

