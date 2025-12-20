# LaTeX-OCR

本專案的目標是製作一個**簡易的 OCR 工具**，能將 **印刷體的數學公式圖片**（如書本掃描或 PDF 擷取圖片）**轉換為 LaTeX code**。本專案不追求高精準度，而是著重於：

* 基本影像處理
* 基本符號辨識
* 基於位置的簡單結構還原

主要挑戰在於：辨識符號、找出上下標與分數等結構。
整體採用 **規則式（rule-based）方法 + 模板比對（template matching）**，不使用深度學習。



## 專案目標

本專案希望達成以下功能：

* 能將單行的基本印刷體公式圖片轉成 LaTeX。
* 支援常見符號（+ − × ÷ = 字母、數字、常見數學符號）。
* 能辨識上下標、分數、根號等基本結構。
* 以模組化方式實作，可日後擴充符號庫。



## 輸入 / 輸出

* **輸入**：一張含單行數學公式的圖片（JPG 或 PNG）。
* **輸出**：一段對應的 LaTeX 字串（例如 `x^2 + \frac{1}{y}`）。



## Pipeline

1. **讀取與預處理**

    實作或使用工具（如 OpenCV）完成：

    * 灰階化
    * 去噪（Gaussian blur）
    * 自適應二值化（adaptive threshold）
    * 找邊緣（Canny）
    * 取得連通區域 bounding boxes（cv2.findContours）

    目的：讓符號輪廓清楚便於後續切割。

2. **符號切割**

    使用 connected components 或 contour bounding box：

    * 找出圖片中所有連通符號的外框（x, y, w, h）
    * 對每個符號進行裁切（crop）

    簡單假設：

    * 相同高度附近的符號為主幹（baseline）
    * 比較高的位置是上標
    * 比較低的位置是下標

    不處理非常複雜的多層級結構。

3. **符號辨識**

    不採深度學習，採用相似度必較（如OpenCV 的 template matching `cv2.matchTemplate`），對每個裁切的符號尋找最像的模板，選擇最佳匹配。

    預期這樣的效果對於印刷體已足夠準確。

4. **結構分析**

    依 bounding box 判斷符號之間的關係：

    - 上下標

        給定主幹高度 H：

        * 如果符號的中心在主幹上方：當作上標 → 使用 `^{}`
        * 如果在主幹下方：當作下標 → 使用 `_{}`
        * 上下標只處理一層，不處理多層巢狀。

    - 分數

        如果偵測到「長水平線」（使用 Hough line 或寬高比判斷）：

        * 線上方區域視為分子
        * 線下方區域視為分母
        * 使用 `\frac{ 分子 }{ 分母 }` 產生輸出

    - 根號

        如果偵測到 root-like 左側符號（template 比對 "\sqrt" 圖片）：

        * 若後面跟著一串符號
        * 則整段包在 `\sqrt{ ... }` 裡
    
    多層結構可考慮遞迴解構。

5. **LaTeX 生成**

    依序將辨識結果組合成 Latex 字串。



## 系統架構

```
input image
    ↓
image preprocessing (OpenCV) 
    ↓
contour detection + symbol cropping  
    ↓
template matching recognition
    ↓
rule-based structural analysis
    ↓
LaTeX generator
    ↓
output LaTeX code
```



## 可能用到的工具

* **OpenCV、PIL**：影像處理、輪廓偵測、模板匹配
* **NumPy**：矩陣運算
* **Python**：主程式語言
* **LaTeX**：用來檢查結果是否能編譯

全部都是基本套件，無需深度學習框架。



<!-- ## Test Cases

1. `x^2`
2. `a + b = c`
3. `\frac{1}{x}`
4. `\sqrt{x+1}`
5. `(x+1)(x-1)`
6. `\alpha^2 + 3\beta` -->



## Milestones

### 第 1–2 週

* 完成圖片預處理
* 完成輪廓偵測與符號切割

### 第 3–4 週

* 完成模板資料庫（20–40 個符號）
* 完成模板匹配辨識

### 第 5 週

* 完成簡單結構（上下標、分數）

### 第 6 週

* 系統整合
* 測試案例集
* 撰寫報告與 demo

## 作業分工
(fill this out)

## 預期限制

* 僅支援印刷體
* 僅支援單行公式
* 不支援過於複雜的公式（如 Power tower, Customized formats）
* 不處理手寫公式
* 旋轉/模糊等情況下表現差

## 未來可擴充方向

* 用深度學習模型（CNN / ViT）取代模板比對
* 自動偵測多行公式
* 使用序列模型（如 Transformer）直接產生 LaTeX
* 加強對根號範圍和括號配對的解析





