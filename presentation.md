# 1. Presentation
## 1.2.1 Introduction
* 組員四人，分工後述
* 城市中有許多種類的聲音，如車聲, 喇叭聲, 施工噪音等等。我們希望能藉由程式識別這些聲音,拓展至更多生活面的應用

## 1.2.2 Methodology
* 訓練模型時使用的輸入為一段長約三秒的聲音檔，使用PCM編碼，但sample rate, bit depth, channel未統一
* 輸出為定義的十種label(警笛,喇叭,引擎怠速,槍響,狗吠,兒童嬉戲,音樂,施工,路面鑽鑿,冷氣運轉)之一

* 模型架構: 
先將聲音檔案轉為mel-scaled spectrogram
resize至512px*512px的圖片 # 圖2
讀入後使用兩層2D Convolution與Pooling
Flatten後使用兩層densely-connected NN將節點收斂至10輸出
\# 圖3
* 訓練完成後使用.save()保存model
* 儲存的model大小約281MB

* 由於希望對應輸出的十個值中一個為1其他為0，loss function選擇使用categorical_crossentropy分類交叉熵函式
optimizer使用SGD。learning rate=0.01, batch size=20, epoch=10, 

## 1.2.3 Dataset
* 訓練資料集共有5435個音檔，大小約4GB。 測試資料集共有3297個音檔，大小約2.6GB
* 取自kaggle的開放資料集Urban Sound Classification(https://www.kaggle.com/pavansanagapati/urban-sound-classification)
* 訓練時使用80%分割，4348筆training sample與1087筆validating sample
另外有3297筆無label的testing sample

## 1.2.4 Experimental Evaluation
* training時使用kaggle kernal,硬體配置如下

CPU: Intel(R) Xeon(R) CPU @ 2.30GHz (2 cores)
RAM: 12.75GB avaliable
GPU: Nvidia Tesla P100-PCIE-16GB (single core)
CUDA: Version 10.1

* 10個epoch後就有過擬合趨勢  # 圖1
* 目前模型準確率最高到0.86 # 圖4
另外由於使用者任意輸入的音檔可能不在原先10個class裡面,設置當所有class的吻合度皆小於閥值時輸出unknown



# 2. SRS Document
## 2.1.1 Purpose
本程式的目的是建立一個能夠準確分析音訊源的模型，以辨別聲音的種類。
應用面上包括:
1. 由於本系統可以辨別槍聲，可以拓展至即時監聽市中心的聲音，防範恐怖攻擊
2. 在國道上設置錄音裝置，辨別汽車引擎聲，觀察數值消長以界定是否塞車
3. 設置於嬰兒房，辨識小孩的哭聲，讓家長能更及時照顧到嬰兒的需求

## 2.1.2 Intended Audience and Reading Suggestion
此項目是音訊源分析及辨識的模型，對於各方面需要分辨聲音或是用聲音辨識做輔助性功能都很有用。
如:城市的管理者、用路人、家長等
## 2.1.3 Project Scope
1. 盡量允許各種格式的音訊源
2. 限制可辨別的音訊類別為事先定義的十種聲音
3. 完成UI介面
4. 防呆機制與例外處理
## 2.2.1 Product Perspective
流程圖
## 2.2.2 Product Functions
1. 註冊系統
2. 登入登出系統
3. admin系統
4. 訊息視窗
5. 主頁面系統
6. 選擇檔案系統
7. 轉換音訊為頻譜圖
8. 預測音訊/頻譜圖的類別並輸出
9. 顯示頻譜圖片於視窗上
10. 顯示頻譜圖片於視窗上
11. 顯示預測結果於視窗上
## 2.2.3 User Classes and Characteristics
1. 一般使用者: 通過圖形化介面操作,可以輕鬆地完成辨識,分析有興趣的聲音
2. 城市管理者: 使用shell script批次實時監測並進行辨識
3. 家長: 架設錄音裝置，於嬰兒哭鬧時可獲得預警訊息
## 2.2.4 Operating Environment
* OS: Windows 10
* Python Runtime: version 3.6
* Packages: run `pip install -r requirements.txt`
* Tensorflow backend
## 2.2.5 Design and Implementation Constraints
訓練模型時由於輸入圖片較大(512*512*1),建議可用GPU RAM在4GB或以上,否則會因頻繁重分配記憶體造成效率低下
由於model已經訓練好,使用者只要在Python環境下，安裝所需套件就可執行
## 2.2.6
需自行安裝的套件(這是一個表格):
1. pandas / 讀取train.csv的label
2. numpy / 將原生list轉為更有效率的numpy.array用於訓練模型
3. PIL / 圖片讀取
4. scipy / 圖片處理
5. matplotlib / 圖片與圖表視覺化呈現
6. librosa / 音訊分析與處理
7. sklearn / 調用one hot encoding, shuffle split等功能
8. tensorflow / keras底下引用的深度學習核心
9. keras / 深度學習的高階API

Python內建的:
1. tkinter / GUI視窗介面
2. pickle / 保存資料
3. os / 讀取檔案

## 2.3 External Interface Requirements

### 2.3.1 User Interfaces
* 使用者執行open_platform_final.py後，開啟tkinter視窗介面，有兩個按鈕: Login / SignUp
* 1.jpg
* 當Password錯誤時會跳出密碼錯誤提示
* 10.jpg
* 當User name錯誤時會跳出註冊提示
* 2.jpg
* 接著進入註冊頁面
* 3.jpg
* 當註冊成功就會跳出提示
* 4.jpg
* 同時會將帳號密碼存入usrs_info.pickle內
* 8.jpg
* 接著登入成功進入主頁面，有兩個按鈕: Choose / LogOut
* 5.jpg
* 點擊Choose選擇要辨識的音源檔
* 6.jpg
* 接著就會出現頻譜以及判斷結果
* 7.jpg
* 點擊LogOut進行登出
* 9.jpg

### 2.3.2 Hardware Interfaces
* 一般電腦即可

### 2.3.3 Softwqre Interfaces
* OS: Windows 10
* Python Runtime: version 3.6
* Packages: run pip install -r requirements.txt
* Tensorflow backend

## 2.4 system features

### 2.4.1 User Registration
2.4.1.1 Description and Priority
	概述為使用者如何去註冊，才可以使用這個系統
	Priority – Medium
2.4.1.2 Stimulus/Response Sequences
系統反應動作使用	    				使用者操作動作
                    				a.使用者想要使用這系統
b.系統回應需要先登入
									c.使用者想登入按下登入按鈕
d.系統回應需要先註冊
									e.使用者想註冊按下註冊按鈕
f.系統彈出註冊視窗
									g.使用者輸入想要帳號密碼並按下確定
h.系統對比資料庫有無重複
i.系統判定有重複並回傳重複訊息給使用者	
									j.使用者重新輸入
k.系統對比資料庫有無重複
l.系統判定無重複並進行註冊	
m.系統回傳註冊成功並跳回登入視窗							

2.4.1.3 Functional Requirements
1. 使用者可以順利的註冊
2. 系統判定使用者輸入的帳密是否有重複
3. 如沒有重複，註冊成功並秀出訊息視窗
4. 當註冊成功，這個帳密可以登入


### 2.4.2 Administration
2.4.2.1 Description and Priority
	概述為在整個系統建立時，就有一個admin的帳號可以使admin不用註冊就可以登入使用系統
	Priority – High
2.4.2.2 Stimulus/Response Sequences
系統反應動作使用	    				使用者操作動作
a.系統在創建時就已創建admin帳密資料
                    				b.使用者想要使用這系統且有admin的帳密
                    				c.使用者使用admin的帳密登入
d.系統判斷admin的帳密是否正確
e.系統判定錯誤並回傳錯誤訊息給使用者	
									f.使用者重新輸入
g.系統判定正確，登入成功並進入主頁面視窗	
									h.使用者開始使用									
An administrator would like to perform updates to users
2.4.2.3 Functional Requirements
1. 在創建系統時，就正確的創建admin的帳密，使監管者或是測試員可以直接登入
2. 可以判斷輸入的帳密是否與admin的帳密一致
3. 當admin的帳密輸入不一致，可以彈出帳密錯誤訊息的視窗
4. 當admin的帳密輸入正確即可以登入進入主頁面視窗

### 2.4.3 Login Logout system
2.4.3.1 Description and Priority
	概述為使用者如何登入和登出系統
	Priority – Medium
2.4.3.2 Stimulus/Response Sequences
系統反應動作使用	    				使用者操作動作
                    				a.使用者想要使用這系統且有註冊成功自己的帳密
                    				b.使用者使用自己的帳密登入
d.系統對比資料庫有無這個帳號
e.系統判定無此帳號並回傳訊息給使用者	
									f.使用者重新輸入
g.系統對比資料庫有無這個帳號
h.系統判定有此帳號並對比密碼是否一致
i.系統判定密碼不一致並回傳訊息給使用者
									j.使用者重新輸入
g.系統對比資料庫有無這個帳號
h.系統判定有此帳號並對比密碼是否一致
i.系統判定密碼一致，登入成功並進入主頁
									j.使用者使用系統
									k.使用者想登出並按下登出按紐								
l.系統接收登出訊息彈出是否登出訊息
									m.使用者按下取消按紐
									n.使用者想登出並按下登出按紐
o.系統接收登出訊息彈出是否登出訊息
									p.使用者按下確定按紐										
q.系統接收登出訊息，登出並回到登入視窗

2.4.3.3 Functional Requirements
1. 可以判定是否有除了admin外的使用者帳密
2. 如沒有除了admin外的使用者帳密，可以切換到註冊系統
3. 可以判斷輸入的帳密是否與資料庫內的帳密一致
4. 當使用者輸入的帳密與資料庫內的帳密不一致，可以彈出帳密錯誤訊息的視窗
5. 當使用者的帳密輸入正確即可以登入進入主頁面視窗
6. 當按下登出按鈕，可以正確的彈出確認訊息視窗
7. 當確認訊息視窗按下取消，可以正確的回到原本畫面
8. 當確認訊息視窗按下確定，可以正確的登出並回到登入系統

### 2.4.4 Main Request System
2.4.4.1 Description and Priority
	概述為主要系統運作，如何去選定要判別的png和wav檔，並輸出結果
	Priority – Very High
2.4.4.2 Stimulus/Response Sequences
系統反應動作使用	    				使用者操作動作
a.登入成功跳轉到主頁面視窗
                    				b.使用者按下choose按鈕
c.系統彈出選擇檔案視窗                  				
                    				c.使用者使用自己的帳密登入
d.系統對比資料庫有無這個帳號
e.系統判定無此帳號並回傳訊息給使用者	
									f.使用者重新輸入
g.系統對比資料庫有無這個帳號
h.系統判定有此帳號並對比密碼是否一致
i.系統判定密碼不一致並回傳訊息給使用者
									j.使用者重新輸入
g.系統對比資料庫有無這個帳號
h.系統判定有此帳號並對比密碼是否一致
i.系統判定密碼一致，登入成功並進入主頁
									j.使用者使用系統
									k.使用者想登出並按下登出按紐								
l.系統接收登出訊息彈出是否登出訊息
									m.使用者按下取消按紐
									n.使用者想登出並按下登出按紐
o.系統接收登出訊息彈出是否登出訊息
									p.使用者按下確定按紐										
q.系統接收登出訊息，登出並回到登入視窗
2.4.4.3 Functional Requirements
1. 按下choose按鈕可以正確地彈出選擇檔案視窗
2. 可以正確的選擇想要的檔案至程式
3. 可以正確的判斷選擇的檔案是音訊或是圖片檔
4. 如不是音訊或是圖片檔，可以輸出找不到檔案到視窗上
5. 可以讀取音訊檔並轉成頻譜圖
6. 可以讀取轉好或預先選定的頻譜圖
7. 可以使用預先訓練好的model分析預測頻譜圖的屬性
8. 可以輸出頻譜圖到視窗上
9. 可以輸出預測結果到視窗上



## 2.5 Other Nonfunctional Requirements
2.5.0 Other Nonfunctional Requirements
1. 清楚的程式架構及簡單明瞭的註解
2. 跨平台相容性
3. 系統需於每月進行pickle維護作業，是否運作正常
4. 系統需於每天檢查是否運作正常
5. 於系統完成後，需撰寫技術文件以方便下一位監管者或修改者使用

2.5.1 Performance Requirements
1. 每次辨識必須在0.1秒完成(讓real-time辨識得以實現)
2. 準確率(預測結果等於實際值)必須在70%以上

2.5.2 Safety Requirements
1. file防呆(只能選定.png或是.wav)
2. 檢查帳號避免重複註冊
3. 檢查帳密是否正確

2.5.3 Security Requirements
1. 使用pickle檔存使用者帳密，pickle檔提供了一個簡單的持久化功能。可以將對象以文件的形式存放在磁盤上，用pickle來序列化使得存使用者帳密不會直接洩漏
