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
## 2.1.2 Intended Audience and Reading Suggestion
此項目是音訊源分析及辨識的模型，對於各方面需要分辨聲音或是用聲音辨識做輔助性功能都很有用。
## 2.1.3 Project Scope
1. 盡量允許各種格式的音訊源
2. 限制可辨別的音訊類別為事先定義的十種聲音
3. 完成UI介面
4. 防呆機制與例外處理
## 2.2.1 Product Perspective
流程圖
## 2.2.2 Product Functions
1. 註冊與登入
2. 轉換音訊為頻譜圖
3. 顯示頻譜圖片於視窗上
4. 預測音訊/頻譜圖的類別並輸出
## 2.2.3 User Classes and Characteristics
1. 無程式能力基礎,想分析手邊聲音資料者: 通過圖形化介面操作,可以輕鬆地完成辨識
2. ...
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
```
2.2.3 要有不同的使用者角色
2.2.6套件可以用表格去做介紹 分別是幹嘛的 不單純只是條列出來
```


## 2.4 system features

### 2.4.1 User Registration
2.4.1.1 Description and Priority
	概述為使用者如何去註冊，才可以使用這個系統
	Priority – Very High
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


### 2.4.2 Administration
2.4.1.1 Description and Priority
	概述為在整個系統建立時，就有一個admin的帳號可以使admin不用註冊就可以登入使用系統
	Priority – Very High
2.4.1.2 Stimulus/Response Sequences
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
2.4.1.3 Functional Requirements

### 2.4.3 Login Logout system
2.4.3.1 Description and Priority
	概述為使用者如何去註冊，才可以使用這個系統
	Priority – Very High
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

### 2.4.4 Main Request System
2.4.4.1 Description and Priority
	概述為使用者如何去註冊，才可以使用這個系統
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



## 2.5 Other Nonfunctional Requirements
2.5.0 Other Nonfunctional Requirements
1. 清楚的程式架構及簡單明瞭的註解
2. 跨平台相容性
2.5.1 Performance Requirements
1. 每次辨識必須在0.1秒完成
2. 準確率(預測結果等於實際值)必須70%以上
3. 
2.5.2 Safety Requirements
1. file防呆(只能選定.png或是.wav)
2. 檢查帳號避免重複註冊
2.5.3 Security Requirements
1. 使用pickle檔存使用者帳密，pickle檔提供了一個簡單的持久化功能。可以將對象以文件的形式存放在磁盤上，用pickle來序列化使得存使用者帳密不會直接洩漏
2. 
