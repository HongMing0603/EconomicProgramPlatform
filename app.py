from flask import Flask, render_template, request, url_for, redirect, session, jsonify, send_from_directory
from flask import g
import mysql.connector
import os
import joblib
import pandas as pd
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import re
import requests
import datetime

from config import goldApi_API_Key


# 導入標準及反標準化Package
sys.path.append('C:\\Users\\user\\OneDrive\\Desktop\\Course\\Seminar\\code\\Data_Normalization_split')
import normalization
from normalization import Denormalize
from normalization import Normalization_afterSplit


app = Flask(__name__)
app.secret_key = os.urandom(24)
# 背景圖片位置
app.static_folder = 'icon'
# 设置静态文件路由
app.static_url_path = '/icon'

# Connect to the database
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='',
    database='login'
)

# Create a table to store user information
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users
    (id INT AUTO_INCREMENT PRIMARY KEY,
     name VARCHAR(255),
     email VARCHAR(255),
     password VARCHAR(255))
''')
print("Connect Database")
conn.commit()


@app.route('/')
def root():
    return render_template('login.html')

# Register page route
@app.route('/register', methods=['GET', 'POST'])
def register():
    # 連接到表單使用這些code
    if request.method == 'POST':
        # Handle form submission
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        # Save as session
        session['email'] = email
        # Add the user to the database or perform other actions here
        cursor.execute('INSERT INTO users (name, email, password) VALUES (%s, %s, %s)',
                       (name, email, password))
        conn.commit()
        return 'Registration successful!'
    else:
        # Show the registration form
        return render_template('register.html')

# Login page router
@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_post():
    username = request.form['username']
    password = request.form['password']
    # Create a database cursor
    mycursor = conn.cursor()
    mycursor.execute("SELECT * FROM users WHERE name = %s AND password = %s", (username, password))
    # Judgement the result exists or not
    result = mycursor.fetchone()
    if result:
        # Store session
        session['name'] = username

        # GET USER ID
        session['user_id'] = result[0]
        # print(session['user_id'])

        
        # Redirect to predict page
        # If you use code = 307 it will sent a post request for page
        # 傳username給html進行顯示
        conn.close()
        return redirect(url_for('predict'), code=307,)
    else:
        # 傳回error變量給html
        # html可以根據判斷有無error變量顯示錯誤訊息
        return render_template('login.html', error='Invalid username or password')


@app.route('/logout')
def logout():
    # 清除session
    session.clear()
    # 改網址
    
    # 導向到login
    return redirect(url_for('login'))

# 個人信息頁面
@app.route('/personalPage')
def personalPage():

    # Connect to the database
    conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='',
    database='login'
)

    # Passing email address
    # email = session['email']

    # Accroding the session id find the user's email address
    mycursor = conn.cursor()
    mycursor.execute("SELECT email FROM users WHERE id = %s",(str(session['user_id']), ))

    # Fetch the result
    result = mycursor.fetchone()
    if result:
        session['email'] = result[0]
        # print(session['email'])
    else:
        print("ID not find")
        

    return render_template('personalPage.html', email = session['email'])



# Prediction Page
# 設置模型資料夾
model_dir = 'C:\\Users\\user\\OneDrive\\Desktop\\Course\\Seminar\\code\\Tune_Parameters\\GridSearchModel\\'

# Page for Predict Select
@app.route('/predict', methods=["POST", "GET"])
def predict():
    username = session.get("name")
    return render_template('Predict.html', username = username)

# @app.route('/predict/BRR', methods=["GET"])
# def predict_BRR():
    # model_name = request.form.get("model_name")
    model_name = request.args.get('model_name')
    print (f"The selected model is: {model_name}")
    # 模型位置
    model = model_dir + model_name
    
    model_file = joblib.load(model)

    # 根據選擇的X 選取相對應的Csv
    # 選擇頁面應該在html選擇傳送選擇的項目到Flask
    # 項目有 gold oil bitcoin
    X_project = ""

    # 導入fillna後的資料
    data = pd.read_csv('C:\\Users\\user\\OneDrive\\Desktop\\Course\\Seminar\\code\\Data\\fillna\\Combine\\fillna_Combine.csv', index_col="Date")
    # 把經濟項目放到最後一個位置
    # Economic_program 要根據fillna combine.csv裡面的經濟項目名稱來取 也就是妳選取的y價格的名稱 
    Economic_program = 'bitcoin-data'
    cols = list(data.columns)
    cols.append(cols.pop(cols.index(f'{Economic_program}_Price')))
    # data用以上index進行排序
    data = data.reindex(columns = cols)

    X = data.iloc[:, :-1]
    y = data.iloc[:,-1:]
    # 分割資料集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Normalization Dataset
    X_train_scalered = Normalization_afterSplit("X_scaler", X_train, "train")
    X_test_scalered = Normalization_afterSplit("X_scaler", X_test, "test")
    y_train_scalered = Normalization_afterSplit("y_scaler", y_train, "train")

    # Convert the scaled data into dataframes
    X_train_df = pd.DataFrame(X_train_scalered, columns=X_train.columns, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_scalered, columns=X_test.columns, index=X_test.index)
    y_train_df = pd.DataFrame(y_train_scalered, columns=y_train.columns, index=y_train.index)
    y_test_df = pd.DataFrame(y_test, columns=y_test.columns, index=y_test.index)

    y_pred = model_file.predict(X_test_df)
    # 調整維度
    y_pred = np.reshape(y_pred, (-1,1))

    # 取得y_test的列名 方便y_pred的列取名
    y_test_col_name = y_test.columns.tolist()[0]
    # 預測完後要反正規化才能與y_test比較
    y_pred = Denormalize(y_pred)
    y_pred_df = pd.DataFrame(y_pred, columns=[f"{y_test_col_name}-Pred"], index=y_test.index)


    # 把y_pred變成json檔
    y_pred_json = y_pred_df.to_json(orient="split")

    # 把y_test變成json檔
    y_test_json = y_test_df.to_json(orient='split',)
    # 傳給chart.html
    # return redirect(url_for('predictions_results', y_test_json = y_test_json))

    return  jsonify(y_test_json = y_test_json,y_pred_json = y_pred_json)


# Plot charts
# @app.route('/predictions_results')
# def predictions_results():
#     return "Hello"
#     print("Plot chart")
#     y_test_json = request.args.get('y_test_json')
      
#     return render_template('test.html', y_test_json = y_test_json) 
    

# @app.route('/predict/PR', methods=["GET"])
# def predict_PR():
    
    model_name = request.args.get('model_name')
    print (f"The selected model is: {model_name}")
    # 模型位置
    model = model_dir + model_name
    model_file = joblib.load(model)
    # 導入fillna後的資料
    data = pd.read_csv('C:\\Users\\user\\OneDrive\\Desktop\\Course\\Seminar\\code\\Data\\fillna\\Combine\\fillna_Combine.csv', index_col="Date")
    # 把經濟項目放到最後一個位置
    cols = list(data.columns)
    Economic_program = 'bitcoin-data'
    cols.append(cols.pop(cols.index(f'{Economic_program}_Price')))
    # data用以上index進行排序
    data = data.reindex(columns = cols)

    X = data.iloc[:, :-1]
    y = data.iloc[:,-1:]
    # 分割資料集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Normalization Dataset
    X_train_scalered = Normalization_afterSplit("X_scaler", X_train, "train")
    X_test_scalered = Normalization_afterSplit("X_scaler", X_test, "test")
    y_train_scalered = Normalization_afterSplit("y_scaler", y_train, "train")

    # Convert the scaled data into dataframes
    X_train_df = pd.DataFrame(X_train_scalered, columns=X_train.columns, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_scalered, columns=X_test.columns, index=X_test.index)
    y_train_df = pd.DataFrame(y_train_scalered, columns=y_train.columns, index=y_train.index)
    y_test_df = pd.DataFrame(y_test, columns=y_test.columns, index=y_test.index)

    y_pred = model_file.predict(X_test_df)
    # 調整維度
    y_pred = np.reshape(y_pred, (-1,1))

    # 取得y_test的列名 方便y_pred的列取名
    y_test_col_name = y_test.columns.tolist()[0]
    # 預測完後要反正規化才能與y_test比較
    y_pred = Denormalize(y_pred)
    y_pred_df = pd.DataFrame(y_pred, columns=[f"{y_test_col_name}-Pred"], index=y_test.index)


    # 把y_pred變成json檔
    y_pred_json = y_pred_df.to_json(orient="split")

    # 把y_test變成json檔
    y_test_json = y_test_df.to_json(orient='split',)
    # 傳給chart.html
    # return redirect(url_for('predictions_results', y_test_json = y_test_json))

    return  jsonify(y_test_json = y_test_json,y_pred_json = y_pred_json)

# @app.route('/predict/EN', methods=["GET"])
# def predict_EN():
    model_name = request.args.get('model_name')
    print (f"The selected model is: {model_name}")
    # 模型位置
    model = model_dir + model_name
    model_file = joblib.load(model)

    # 根據選擇
    # 導入fillna後的資料
    data = pd.read_csv('C:\\Users\\user\\OneDrive\\Desktop\\Course\\Seminar\\code\\Data\\fillna\\Combine\\fillna_Combine.csv', index_col="Date")
    # 把經濟項目放到最後一個位置
    Economic_program = 'bitcoin-data'
    cols = list(data.columns)
    cols.append(cols.pop(cols.index(f'{Economic_program}_Price')))
    # data用以上index進行排序
    data = data.reindex(columns = cols)

    X = data.iloc[:, :-1]
    y = data.iloc[:,-1:]
    # 分割資料集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Normalization Dataset
    X_train_scalered = Normalization_afterSplit("X_scaler", X_train, "train")
    X_test_scalered = Normalization_afterSplit("X_scaler", X_test, "test")
    y_train_scalered = Normalization_afterSplit("y_scaler", y_train, "train")

    # Convert the scaled data into dataframes
    X_train_df = pd.DataFrame(X_train_scalered, columns=X_train.columns, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_scalered, columns=X_test.columns, index=X_test.index)
    y_train_df = pd.DataFrame(y_train_scalered, columns=y_train.columns, index=y_train.index)
    y_test_df = pd.DataFrame(y_test, columns=y_test.columns, index=y_test.index)

    y_pred = model_file.predict(X_test_df)
    # 調整維度
    y_pred = np.reshape(y_pred, (-1,1))

    # 取得y_test的列名 方便y_pred的列取名
    y_test_col_name = y_test.columns.tolist()[0]
    # 預測完後要反正規化才能與y_test比較
    y_pred = Denormalize(y_pred)
    y_pred_df = pd.DataFrame(y_pred, columns=[f"{y_test_col_name}-Pred"], index=y_test.index)


    # 把y_pred變成json檔
    y_pred_json = y_pred_df.to_json(orient="split")

    # 把y_test變成json檔
    y_test_json = y_test_df.to_json(orient='split',)
    # 傳給chart.html
    # return redirect(url_for('predictions_results', y_test_json = y_test_json))

    return  jsonify(y_test_json = y_test_json,y_pred_json = y_pred_json)

# 預測項目及模型傳入
@app.route('/predict/<model>/<economic_variable>', methods=["GET"])
def predict_model_variable(model, economic_variable):
    print(model, economic_variable)

    # 讀取模型:
    model_location = f"code/Data/GridSearchModel/{model}_{economic_variable}"
    modelfile = joblib.load(model_location)
    print(modelfile)

    # 讀取數據
    data = pd.read_csv('code/Data/fillna/Combine/fillna_Combine.csv', index_col="Date")

    # 使用正則表達式取出經濟項目名稱
    string = economic_variable
    pattern = r"All2(.*)"

    match = re.search(pattern, string)

    if match:
        economic_variable = match.group(1)
    else:
        print("No match found.")
    cols = list(data.columns)
    cols.append(cols.pop(cols.index(f'{economic_variable}')))
    # data用以上index進行排序
    data = data.reindex(columns = cols)

    # 判斷是否有High 與 Low 如果有的話就丟棄(無效預測)
    
    economic_name = economic_variable.split('_')[0]
    drop_columns = [f"{economic_name}_High", f"{economic_name}_Low"]
    for drop_column in drop_columns:
        try:
            data = data.drop(drop_column, axis=1)
        except:
            pass
    X = data.iloc[:, :-1]
    y = data.iloc[:,-1:]
    # 分割資料集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Normalization Dataset
    X_train_scalered = Normalization_afterSplit("X_scaler", X_train, "train")
    X_test_scalered = Normalization_afterSplit("X_scaler", X_test, "test")
    y_train_scalered = Normalization_afterSplit("y_scaler", y_train, "train")

    # Convert the scaled data into dataframes
    X_train_df = pd.DataFrame(X_train_scalered, columns=X_train.columns, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_scalered, columns=X_test.columns, index=X_test.index)
    y_train_df = pd.DataFrame(y_train_scalered, columns=y_train.columns, index=y_train.index)
    y_test_df = pd.DataFrame(y_test, columns=y_test.columns, index=y_test.index)

    y_pred = modelfile.predict(X_test_df)
    # 調整維度
    y_pred = np.reshape(y_pred, (-1,1))

    # 取得y_test的列名 方便y_pred的列取名
    y_test_col_name = y_test.columns.tolist()[0]
    # 預測完後要反正規化才能與y_test比較
    y_pred = Denormalize(y_pred)
    y_pred_df = pd.DataFrame(y_pred, columns=[f"{y_test_col_name}-Pred"], index=y_test.index)


    # 把y_pred變成json檔
    y_pred_json = y_pred_df.to_json(orient="split")

    # 把y_test變成json檔
    y_test_json = y_test_df.to_json(orient='split',)
    # 傳給chart.html
    # return redirect(url_for('predictions_results', y_test_json = y_test_json))

    return  jsonify(y_test_json = y_test_json,y_pred_json = y_pred_json)


# Home page
@app.route('/home')
def home():
    return render_template('home_page.html', username = session['name'])

# about page
@app.route('/about')
def about():
    return render_template('about.html', username = session['name'])

# favorite page 
@app.route('/favorites')
def favorites():
    return render_template('favorite.html', username = session['name'])




# Read css file
@app.route('/css/<file>')
def serve_css(file):
    return send_from_directory('static/css', file)

# Read javascript file
@app.route('/javascript/<path:filename>')
def serve_scripts(filename):
    # get user name
    username = session['name']
    return send_from_directory('static/javascript', filename)
# Get on time price 
@app.route('/onTimePrice', methods=['GET'])
def get_on_time_price():
    # Make a GET request to the CoinGecko API for Bitcoin data
    response = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd')

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()

        # Get the Bitcoin price from the response
        bitcoin_price = data.get('bitcoin', {}).get('usd')

        # Return the Bitcoin price as a JSON response
        return jsonify(price=bitcoin_price)
    else:
        # Return an error message as a JSON response
        return jsonify(error='Failed to retrieve Bitcoin price')
    
def get_gold_price():
    # result is one troy ounce approximately 31.1034768 g
    url = 'https://www.goldapi.io/api/XAU/USD'
    headers = {
        'x-access-token': goldApi_API_Key  # Replace with your Gold-API API key
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        gold_price = data['price']
        gold_timestamp = data['timestamp']
        gold_datetime = datetime.datetime.utcfromtimestamp(gold_timestamp).strftime('%Y-%m-%d %H:%M:%S')
        return gold_price, gold_datetime
    return None, None

@app.route('/economic_prices')
def economic_prices():
    response = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd')
    bitcoin_data = response.json()
    bitcoin_price = bitcoin_data['bitcoin']['usd']

    bitcoin_info = requests.get('https://api.coindesk.com/v1/bpi/currentprice/BTC.json')
    bitcoin_info_data = bitcoin_info.json()
    bitcoin_timestamp = bitcoin_info_data['time']['updated']
    gold_price, gold_timestamp = get_gold_price()
    
    print(gold_timestamp)

    data = {
        'Bitcoin': {
            'value': bitcoin_price,
            'timestamp': bitcoin_timestamp
        },
        'Gold': {
            'value': gold_price,
            'timestamp': gold_timestamp
        },
        # Add more data as needed
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, port=5666)
    

