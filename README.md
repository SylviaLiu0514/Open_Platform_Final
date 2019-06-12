Open Platform Final
====
# load_image.py
## 讀spectrogram資料夾內的所有頻譜(.png)
# read_csv_data.py
## 讀train.csv的資料和load_image.py讀進的圖片放一起
# training.py
合併其他子function
運行後訓練model並保存為urban_sound.h5
# user interface cmd版
1. 確保predict.py與預先訓練好的model(urban_sound.h5)於相同路徑
2. 運行predict.py
3. 輸入欲預測的檔案名稱
4. 輸入enter結束程式

# user interface tkinter版
1. 確保predict.py與預先訓練好的model(urban_sound.h5)於相同路徑
2. 運行open_platform_final.py
3. 頁面開啟有兩個button
4. 一個是Login，之前已註冊(不管有無重新開啟)，登入，p.s.因會存.pickle檔，故有防資料洩漏
5. 一個是SignUp，沒帳戶註冊
6. 進到新頁面有兩個button
7. 一個是Choose，點了會跳出視窗請你選你的圖檔(.png)或是音檔(.wav)
8. 選完後會顯示轉成頻譜的圖片和maching learning判斷的事物
9. 一個是Logout，點了登出回到登入的視窗
