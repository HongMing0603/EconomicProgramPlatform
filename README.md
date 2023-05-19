## 製作項目
1. 個人資訊頁面
2. 新增預測介面功能

## 會員個人介面
![picture 1](images/22d61e4c0395b99b7954e735aa058971d67bcc4328549d1da5518cc207c09cda.png)  

目前還不能直接顯示會員的個人訊息，正在嘗試從資料庫當中獲取數據顯示於此頁面

### 更新
![picture 7](images/fe6298b96dccb1886bc7007de062ae58aa2d79e5a619582056e8d7d188bf6a20.png)  
已經可以顯示使用者的姓名及電子信箱位置，但還不能更改資料庫

## 預測介面
![picture 3](images/a5f7ef4692962820f1d33767dcb8ea2bcca630e5ff7763269a43a5b1f57af312.png)  

- 增加了選擇預測項目的功能，使用者可以選擇一經濟項目進行預測。

![picture 4](images/f71547af1368d290efe085f3900d730017b44a345e110a33e7bfcb9624757f78.png)  

## 持續更新中:
![picture 5](images/eead81871c1255d6153318d893a7f22ff32bc0f3acd53912e86c419ff4c0b1b3.png)  

![picture 6](images/83d9aaede4f89a1d67ea5aba11e3b638fa037b2bef7b9bafde3d7e436a0ce49e.png)  

# 使用教學
## 配置環境
Anaconda官方教學:
> https://www.anaconda.com/blog/moving-conda-environments

Cd至(根目錄)於Terminal當中輸入

```conda env create -f environment.yml```

即可安裝環境

本專案使用**VScode**製作

於VScode切換到對應環境
![picture 8](images/e70f28a60c691383f1ce8c1fb3c306f4b7f9a59d99e0da293c033faf0f628bd1.png)  

## 安裝Xampp

#### Xampp configuration:
![picture 9](images/943526b8d9c44aa85b36011f5c5d33e44c20ebb121fed4a8f1d29064dc72bc5c.png)  
> 修改自相同的port

### import sql file
From Xampp control panel
![picture 10](images/53a87084028b6e7d3b8ea4f83195d9a751ec9b77db2a77ef69da0ba1a33a71c3.png)  
可以進入到phpMyadmin介面

按下匯入
![picture 11](images/d5479fbad10f57602189483305fbc6f53bcd1dc73f5c0dabc1c4ff103b98efec.png)  

按下選擇檔案
![picture 12](images/d9fb047f0ac2ebaefddd1670b571e9c29cd5e5f783d54e58cd8a7ce151066d83.png)  

於本專案資料夾根目錄可看到sql檔案
![picture 13](images/b37efed6da4a4ccb3401a4751d37cafb76e6c1dea41b3988ab860850168fe3e1.png)  

點選127_0_0_1.sql
按開啟

![picture 14](images/f846732a4c857c9716638e776ec2c879c5896dcc4db8901f459e0e23065fdf63.png)  

將進度調移動到最下方可以看到匯入按鈕
點擊它即可匯入資料

### 運行app.py即可搭建預測平台



