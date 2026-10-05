# 高速公路流量預測 (Highway Traffic Flow Prediction)

整合國道電子收費（ETag）車流資料與氣象降雨資料，建立並比較 LSTM / GRU 模型，對高速公路路段車流量進行時間序列預測。

## 技術棧
- Python, pandas（資料清理、合併）
- TensorFlow / Keras（LSTM、GRU）
- scikit-learn（標準化、評估指標）

## 主要檔案
- `LSTM_n_GRU.py`：LSTM / GRU 模型定義、訓練與評估（示範流程）
- `upload/dataset.py`：讀取並合併每日高速公路 M05A 車流資料與對應測站降雨量，輸出月資料（`merged/`）與各日執行時間記錄（`execution_time/`）
- `upload/lstm.py`～`lstm4.py`：不同版本的 LSTM 預測模型（含以經典 airline-passengers 資料集進行的原型測試）
- `upload/missing.py`：補齊合併資料中缺漏日期
- `upload/*.png`、`*.jpg`：各次模型調整過程的訓練結果與除錯截圖

## 資料集
- **高速公路局 M05A 國道電子收費（ETag）車流資料**：交通部高速公路局公開之路段別車種通行量與平均車速資料（5 分鐘區間），本 repo 僅保留處理程式，未包含原始/合併後的 CSV（約 80MB，`upload/merged/`、`upload/execution_time/`、`upload/n7_temp/`）。
- **中央氣象署測站降雨量資料**：對應路段附近測站之逐時降雨量，用於合併為車流資料的額外特徵。
- `upload/lstm.py` 中另有一段以公開示範資料集 `international-airline-passengers.csv`（經典航空乘客月資料）進行的模型原型測試，與主要高速公路資料無關。

## 執行方式
```bash
pip install pandas tensorflow scikit-learn matplotlib
python upload/dataset.py   # 產生合併後的月資料
python LSTM_n_GRU.py       # 訓練並評估 LSTM/GRU 模型
```
