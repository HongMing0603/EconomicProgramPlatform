
# Contributor!!!
CoinDaddy0503
# 錢老爺經濟項目平台

![picture 1](images/8b9ebfc5e597197d58afafaae09b2ba6b807248be325b8dee8a5ebff8329cd58.png)  


## 配置環境
Anaconda官方教學:
> https://www.anaconda.com/blog/moving-conda-environments

Cd至(根目錄)於Terminal當中輸入

```conda env create -f environment.yml```

即可安裝環境

## Abstract
自從20世紀初內燃機技術的迅速發展，作為其燃料的**石油地位不斷的提升**，至今已成為世界上重要的能源之一。基於石油對於整個世界有著崇高的地位，本專題想要藉此機會來探討石油價格變化的趨勢。加密貨幣是基於區塊鏈技術的新型交易媒介，近年以比特幣及以太幣等主流加密貨幣，對全球商業經濟的影響力逐漸增長，受到政府部會及產業界的重視。分析金融經濟議題的典型方法為整合移動平均自回歸模型（Autoregressive Integrated Moving Average model）時間序列分析方法，然而，對於近年快速變化的複雜金融商業環境，時間序列分析方法的預測能力仍有提昇空間。本專題將探討3種時間序列分析方法、3種機器學習方法、2種時空分析方法，並整合建構為一經濟時空分析平台，進行油價、黃金價格等2種經濟指標對於加密貨幣的趨勢預測效能分析。本專題成果將提供一種經濟時空分析的解決方案，能夠進一步理解經濟指標和加密貨幣的影響及交互關係，並探討經濟指標在時間與空間上的不同效果，以供政府部會及產業擬定政策參考。


## Tutorial
使用者可以執行 Data_Combination 資料夾下的CombinationTrain.py 就能夠使用各種經濟項目組合並得到經濟項目組合在各模型下的評估結果

各程式描述於下方說明。


---
## Code Parsing
---

### Bitcoin_data_download_api(Folder)

#### DataProcessing .py
用於下載加密貨幣，目前可以選擇比特幣或是以太幣，下載的資料欄位為Open, High, Low
Price 為手工處理欄位，為(High + Low) /2
最後將下載的資料放到Data/crypto_downloader/裡面

### Data (Folder)
用於儲存處理過的csv文件
- Combination (存放各種經濟項目組合的csv，ex:oil, gold, gold + oil, bitcoin + gold + oil ....)
- crypto_downloader (存放從DataProcessing.py 下載的加密貨幣資料)
- fillna (存放填補缺失值後的各種經濟項目資料)
  - Combine (將所有的經濟項目組合到一個csv文件當中，以後可以設計成，選取一些想要的經濟項目就從裡面的csv文件選取，而不用導入多個csv文件?)
- Normalization (存放標準化的經濟項目資料)
  - split_to_XY (分為訓練集的標準化與測試集的標準化，名字的格式為 經濟項目2預測項目-data-train(test).csv)

### RangeChange (Folder)
存放著調整過資料期間的csv文件，調整日期是為了保證資料的一致性，不會有太多的缺失值。調整完資料期間後就可以拿著這些資料去進行資料缺失值的填充

### Data_Combination (Folder)
- CombinationTrain.py (從 Data/combination讀取csv文件)直接選取一個文件(現在是手動設置，當作y)遍歷每一種經濟項目Combination對y進行預測，預測時使用5種機器學習算法，config 用來儲存 X變量與Y變量名稱，方便命名
  
- EconomicCombination.py (用於創造各種經濟項目組合的csv文件)
  - 實現方式: 讀取fillna的csv fillna資料夾有各個經濟項目fillna後的csv檔，讀取每個csv檔後，把個別經濟項目的 Price, Open, High, Low 存入到一個DataFrame當中，並且輸出為csv文件(輸出為 Oil, Gold, Gold + Oil ... )
  

### Data_fillna (Folder)
- fillna.py (從Data/RangeChanges裡面讀取csv經濟項目文件，並且填充缺失的部分，最後輸出到Data\\fillna\\Combine\\ 裡面)

### Data_Normalization_split (Folder)
- normalization.py (用來標準化經濟項目資料)

### Data_range_change (Folder)
- rangeChange.py (修改原始數據的資料期間現在設定為 2015-01-01 ~ 2023-01-23 為了實驗方便用 我們有設定一個變量，你可以輸入一些想要刪除的欄位，之後數據就只保留想要進行分析的欄位)

### economic_project (Folder)
用於存放經濟項目原始資料

### Train_Prediction (Folder)
放置會使用到的模型，其實好像可以把這些模型全部放到一個py文件當中就好?

### Tune_Parameters (Folder)
- BayesSearchModel (用來放置各個經濟項目經過Bayes Search 後的模型)
- GridSearchModel (用來放置各個經濟項目經過Grid Search 後的模型)
- BayesSarch.py (Bayes 搜索)
- GridSearch.py (GridSearch 搜索)
- SplitData.py (用來分割訓練測試集，實驗用定為0.2方便數據製作成表格)
  
### Validation_index (Folder)
- vd_index.py (有RMSE, MAPE, SMAPE,MAE, R2五種指標，使用著可以傳入y_test,與y_pred 讓他計算評估指標)

