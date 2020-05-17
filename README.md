
# 基於機器學習之聲音辨識系統

我們開發了基於機器學習之聲音辨識系統，本程式的目的是建立一個能夠準確分析音訊源的模型，以辨別聲音的種類。

我們用機器學習的訓練，使得我們先對每個聲音轉成頻譜圖，再以每個頻譜圖做類別區分
，之後使用此model為之後使用者選定的聲音或是圖片檔做辨識判定，
呈現頻譜圖和判定結果到視窗給使用者觀看。



## 訓練過程


訓練模型時使用的輸入為一段長約三秒的聲音檔，使用PCM編碼，但sample rate, bit depth, channel未統一

輸出為定義的十種label之一
包含:警笛,喇叭,引擎怠速,槍響,狗吠,兒童嬉戲,音樂,施工,路面鑽鑿,冷氣運轉

### 模型架構

先將聲音檔案轉為mel-scaled spectrogram resize至512px512px的圖片
讀入後使用兩層2D Convolution與Pooling
Flatten後使用兩層densely-connected NN
將節點收斂至10輸出

由於希望對應輸出的十個值中一個為1其他為0，
loss function選擇使用categorical crossentropy分類交叉熵函式，optimizer使用SGD

### 使用資料
資料取自kaggle的開放資料集Urban Sound Classification
訓練資料集共有5435個音檔，大小約4GB
其中4348筆為training sample ，1087筆為validating sample
測試資料集共有3297個音檔，大小約2.6GB


10個epoch後就有過擬合趨勢

![](https://i.imgur.com/ENIRf8i.png)


目前模型準確率最高到0.86
另外由於使用者任意輸入的音檔可能不在原先10個class裡面,設置當所有class的吻合度皆小於閥值時輸出unknown

## 訓練結果

我們實測實測5個來源、長度各異的音檔，準確率80%


| 實際值| 辨識值 |
| -------- | -------- |
| dog_bark     | dog_bark     |
| drilling     | unknow     |
| engine_idling     | engine_idling     |
| siren     | siren     |
| children_playing     | children_playing     |


## 使用方式

使用者執行open_platform_final.py後，開啟tkinter視窗介面，有兩個按鈕供註冊以及登入

![](https://i.imgur.com/WK4918p.jpg)

進入主畫面以後供選擇檔案，可以選擇音檔或者圖片檔，如果選擇音檔會將該音檔轉換成頻譜圖，並放進訓練好模型裡做分析
![](https://i.imgur.com/Fy1D6rh.png)




接著就會出現頻譜以及判斷結果

![](https://i.imgur.com/LWjK4Nc.jpg)
