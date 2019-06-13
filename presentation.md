# 1. Presentation
## 1.2.1 Introduction
* 組員四人，分工後述
* 城市中有許多種類的聲音，如車聲, 喇叭聲, 施工噪音等等。我們希望能藉由程式識別這些聲音,拓展至更多生活面的應用

## 1.2.2 Methodology
* 訓練模型時使用的輸入為一段長約三秒的聲音檔，使用PCM編碼，但sample rate, bit depth, channel未統一
* 輸出為定義的十種label(警笛,喇叭,引擎怠速,槍響,狗吠,兒童嬉戲,音樂,施工,路面鑽鑿,冷氣運轉)之一

* 模型架構:
先將聲音檔案轉為mel-scaled spectrogram
resize至512px*512px的灰階圖片
讀入後使用兩層2D Convolution與Pooling
Flatten後使用兩層densely-connected NN將節點收斂至10輸出

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

* 10個epoch後就有過擬合趨勢  # 這邊要附上圖片
* 目前模型準確率0.75
另外由於使用者任意輸入的音檔可能不在原先10個class裡面,設置當所有class的吻合度皆小於閥值時輸出unknown



# 2. SRS Document
## 2.1.1 Purpose
目的
## 2.1.2 Intended Audience and Reading Suggestion
軟體受眾
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
使用者為任何想分析手邊聲音資訊者
通過圖形化介面操作,不需要程式能力基礎
## 2.2.4 Operating Environment
OS: Windows 10
Python Runtime: version 3.6
Packages: run `pip install -r requirements.txt`
Tensorflow 
## 2.2.5 Design and Implementation Constraints
訓練模型時由於輸入圖片較大(512*512*1),建議可用GPU RAM在4GB或以上,否則會因頻繁重分配記憶體造成效率低下
由於model已經訓練好,使用者只要在Python環境下，安裝所需套件就可執行
## 2.2.6
需自行安裝的套件:
1. pandas
2. numpy
3. scipy
4. tensorflow
5. librosa
6. matplotlib
7. keras
8. sklearn
9. PIL

```
2.2.3 要有不同的使用者角色
2.2.6套件可以用表格去做介紹 分別是幹嘛的 不單純只是條列出來
```
