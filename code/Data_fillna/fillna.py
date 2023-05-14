import pandas as pd
import glob
import os
import datetime

# fillna Package
from sklearn.impute import SimpleImputer

# Define the Economic_list
economic_column = ["Price", "Open", "High", "Low"]

files = glob.glob('Data\RangeChanges\*.csv')

def fillna():

    for file in files:
        df = pd.read_csv(file)
        # set the Date column
        df = df.rename(columns={df.columns[0]:'Date'})
        file_name = os.path.basename(file)
        file_name = file_name.replace(".csv", "")
        # set DataFrame index
        df = df.set_index("Date")
        print(f"Now file is {file_name} ")
        print(df.isnull().sum())

        # Use SimpleImputer fillna
        imputer = SimpleImputer(strategy="mean")
        imputed_data = imputer.fit_transform(df)

        # Create a DataFrame
        imputed_df = pd.DataFrame(imputed_data, columns=df.columns, index=df.index)

        directory_name = "Data/fillna"

        # Create a fillna Directory
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
        else:
            print("Directory alreay exists")

        # Create to csv
        imputed_df.to_csv(f'{directory_name}/{file_name}_fillnaData.csv', index=True)
        # Check dataFrame number of isnull
        print(imputed_df.isnull().sum().sum())
        
    
# 得到最新的日期
date = datetime.datetime.today().strftime("%Y-%m-%d")
# Create a DataFrame for All-fillna files
time_index = pd.date_range("2015-01-01", date, freq='D')
df = pd.DataFrame(index = time_index)

def FillnaCombine():
    """
    Combine for each fillna csv
    Output -> DataFrame = Combine fillna data
    
    """
    # Combine from various CSVS
    files = glob.glob('Data\\fillna\\*.csv')
    # 先設定資料最新的時間方便測試
    # 取得資料的最後一天避免創造的DataFrame錯亂
    file = 'Data\\fillna\\bitcoin-data_RangeChanges_fillnaData.csv'
    reader = pd.read_csv(file)
    # convert the date column to datetime format
    reader['Date'] = pd.to_datetime(reader['Date'])

    # sort the DataFrame by date in descending order
    reader.sort_values(by='Date', ascending=False, inplace=True)

    # select the first row (which has the latest date) and extract the date
    last_date = reader.iloc[0]['Date'].date()        
    time_index = pd.date_range("2015-01-01", last_date, freq='D')
    df = pd.DataFrame(index = time_index)

    # input data into dataframe and rename for column
    for file in files:
        

        # In here file name is the economic_program's name
        file_name = os.path.basename(file)
        file_name = file_name.split("_")[0]
        # Read the data so that it can fit into ours new DataFrame
        file_df = pd.read_csv(file)
        # Determine if the fields in the csv file are what we want
        # Take out the individual column
        for column in file_df.columns:
            if column in economic_column:
                # put data into ours new dataframe
                df.loc[:, f'{file_name}_{column}'] = file_df[column].values

    # Output this complete DataFrame to csv
    df.index.name = 'Date'
    df.to_csv('Data\\fillna\\Combine\\fillna_Combine.csv', index=True)
    # Return the dataFrame
    return df

if __name__ == "__main__":
    FillnaCombine()