# test.py 模型測試腳本說明文件

## 概述
`test.py` 是 YOLO-PCB 專案中的模型測試腳本，用於在測試集上評估訓練好的 YOLOv5 模型效能。提供詳細的評估指標和分析結果，是模型驗證的重要工具。

## 主要功能

### 1. 模型評估流程

#### 1.1 測試準備階段
- **模型載入**：載入訓練好的權重檔案
- **資料集準備**：載入測試集影像和標籤
- **設備配置**：自動選擇 CPU/GPU 進行推論
- **評估設定**：配置信心度閾值和 IoU 閾值

#### 1.2 批次推論階段
```python
# 對每個批次執行推論
for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
    # 前向傳播
    inf_out, train_out = model(img, augment=augment)
    # 非極大值抑制
    output = non_max_suppression(inf_out, conf_thres, iou_thres)
    # 計算評估指標
    stats.append(statistics_per_image)
```

#### 1.3 指標計算階段
- **精確率與召回率**：針對不同 IoU 閾值計算
- **平均精度 (AP)**：計算每個類別的 AP 值
- **mAP 計算**：計算整體和各類別的 mAP
- **混淆矩陣**：分析類別間的混淆情況

### 2. 核心評估指標

#### 2.1 Average Precision (AP)
- **AP@0.5**：IoU 閾值為 0.5 時的平均精度
- **AP@0.5:0.95**：IoU 閾值從 0.5 到 0.95（步長 0.05）的平均精度
- **各類別 AP**：每個 PCB 缺陷類別的獨立評估

#### 2.2 Mean Average Precision (mAP)
```python
# mAP@0.5 計算
map50 = ap[:, 0].mean()
# mAP@0.5:0.95 計算  
map = ap.mean()
```

#### 2.3 Precision & Recall
- **Precision (精確率)**：TP / (TP + FP)
- **Recall (召回率)**：TP / (TP + FN)
- **F1-Score**：2 * (Precision * Recall) / (Precision + Recall)

#### 2.4 混淆矩陣
- 分析模型在不同類別間的混淆情況
- 識別容易誤分的類別組合
- 評估模型的類別區分能力

### 3. 測試模式支援

#### 3.1 驗證模式 (val)
```bash
python test.py --data PCB.yaml --weights best.pt --task val
```
- 在驗證集上評估模型效能
- 用於訓練過程中的模型選擇

#### 3.2 測試模式 (test)
```bash
python test.py --data PCB.yaml --weights best.pt --task test
```
- 在測試集上進行最終評估
- 提供模型的客觀效能指標

#### 3.3 批次處理
- 支援單張影像或批次影像測試
- 自動調整批次大小以優化記憶體使用

## 命令列參數

### 基本參數
- `--data`：資料集配置檔案（PCB.yaml）
- `--weights`：模型權重檔案路徑
- `--batch-size`：測試批次大小（預設：32）
- `--img-size`：測試影像尺寸（預設：640）
- `--task`：測試任務（'val' 或 'test'）

### 評估參數
- `--conf-thres`：信心度閾值（預設：0.001）
- `--iou-thres`：NMS 的 IoU 閾值（預設：0.6）
- `--augment`：啟用測試時增強（TTA）
- `--verbose`：顯示詳細結果
- `--single-cls`：將所有類別視為單一類別

### 輸出參數
- `--save-txt`：保存檢測結果為文字檔案
- `--save-conf`：在文字檔案中包含信心度分數
- `--save-json`：保存 COCO 格式的結果
- `--project`：結果保存專案目錄
- `--name`：實驗名稱

## 使用範例

### 1. 基本測試
```bash
python test.py --data PCB.yaml --weights ./weights/best.pt --img-size 608
```

### 2. 詳細評估
```bash
python test.py --data PCB.yaml --weights ./weights/best.pt --batch-size 1 --verbose --save-txt
```

### 3. 測試時增強 (TTA)
```bash
python test.py --data PCB.yaml --weights ./weights/best.pt --augment --img-size 608
```

### 4. 指定信心度閾值
```bash
python test.py --data PCB.yaml --weights ./weights/best.pt --conf-thres 0.5 --iou-thres 0.45
```

### 5. 保存 COCO 格式結果
```bash
python test.py --data PCB.yaml --weights ./weights/best.pt --save-json --project runs/test --name pcb_eval
```

## 評估結果解析

### 1. 終端輸出格式
```
                 Class     Images     Targets           P           R      mAP@.5  mAP@.5:.95
                   all        534        1338       0.892       0.851       0.895       0.623
           missing_hole        534         234       0.898       0.863       0.911       0.645
            mouse_bite        534         189       0.885       0.847       0.889       0.612
          open_circuit        534         267       0.901       0.856       0.902       0.634
                 short        534         198       0.889       0.845       0.887       0.608
                  spur        534         223       0.896       0.852       0.898       0.625
       spurious_copper        534         227       0.883       0.843       0.881       0.615
```

### 2. 指標說明
- **Class**：PCB 缺陷類別名稱
- **Images**：測試影像總數
- **Targets**：該類別的真實標註數量
- **P (Precision)**：精確率
- **R (Recall)**：召回率
- **mAP@.5**：IoU=0.5 時的平均精度
- **mAP@.5:.95**：IoU 0.5-0.95 的平均精度

### 3. 各類別詳細分析

#### 缺陷類別對應表
| 類別 ID | 中文名稱 | 英文名稱 | 特徵描述 |
|---------|----------|----------|----------|
| 0 | 缺孔 | missing_hole | 圓形或橢圓形缺失 |
| 1 | 鼠咬 | mouse_bite | 邊緣鋸齒狀缺陷 |
| 2 | 斷路 | open_circuit | 線路中斷 |
| 3 | 短路 | short | 意外連接 |
| 4 | 毛刺 | spur | 細小突起 |
| 5 | 偽銅 | spurious_copper | 多餘銅箔 |

## 結果檔案輸出

### 1. 檔案結構
```
runs/test/exp/
├── labels/              # YOLO 格式檢測結果
│   ├── image1.txt
│   └── image2.txt
├── test_batch0_labels.jpg   # 真實標籤可視化
├── test_batch0_pred.jpg     # 預測結果可視化
├── confusion_matrix.png     # 混淆矩陣圖
├── results.txt             # 詳細數值結果
└── predictions.json        # COCO 格式結果（若啟用）
```

### 2. 文字檔案格式 (YOLO)
```
# 格式：class_id center_x center_y width height [confidence]
0 0.742 0.396 0.186 0.142 0.892
1 0.315 0.628 0.094 0.076 0.856
```

### 3. JSON 檔案格式 (COCO)
```json
[
  {
    "image_id": 1,
    "category_id": 0,
    "bbox": [258.15, 41.29, 348.26, 243.78],
    "score": 0.892
  }
]
```

## 效能分析工具

### 1. 混淆矩陣分析
```python
# 分析類別間混淆情況
confusion_matrix = ConfusionMatrix(nc=6)
# 生成混淆矩陣圖表
confusion_matrix.plot(save_dir=save_dir, names=names)
```

### 2. PR 曲線分析
- **Precision-Recall 曲線**：評估不同閾值下的效能
- **各類別獨立分析**：識別表現較差的類別
- **閾值選擇指導**：為實際應用選擇最佳閾值

### 3. 速度測試
```bash
# 測量推論速度
python test.py --data PCB.yaml --weights best.pt --batch-size 1 --device 0
```

## 高級功能

### 1. 測試時增強 (TTA)
```bash
# 啟用 TTA 提高檢測精度
python test.py --data PCB.yaml --weights best.pt --augment
```
- 多尺度測試
- 翻轉增強
- 結果平均化

### 2. 自動標註功能
```bash
# 生成偽標籤用於半監督學習
python test.py --data PCB.yaml --weights best.pt --save-txt --save-conf --conf-thres 0.8
```

### 3. 批次效能評估
```python
# 在 Python 中直接調用
from test import test

results = test(
    data='PCB.yaml',
    weights='best.pt',
    batch_size=32,
    imgsz=608,
    conf_thres=0.001,
    iou_thres=0.6,
    save_json=True,
    verbose=True
)
```

## 效能基準

### 1. 標準評估設定
- **影像尺寸**：608x608
- **批次大小**：32
- **信心度閾值**：0.001
- **IoU 閾值**：0.6

### 2. 預期效能指標
```
# 基於 baseline_fpn_loss.pt 權重
mAP@0.5: 0.89-0.91
mAP@0.5:0.95: 0.62-0.65
推論速度: ~5-10ms/image (RTX3070)
```

## 故障排除

### 1. 常見錯誤
```bash
# 資料集路徑錯誤
FileNotFoundError: [Errno 2] No such file or directory: '../PCB_dataset/images/testPCB'
# 解決：檢查 PCB.yaml 中的路徑設定

# 記憶體不足
RuntimeError: CUDA out of memory
# 解決：減少批次大小 --batch-size 16

# 權重檔案不相容
KeyError: 'model'
# 解決：確認使用正確的權重檔案
```

### 2. 效能問題
- **推論速度慢**：使用 GPU、啟用半精度
- **記憶體使用過多**：減少批次大小
- **精度不理想**：檢查資料品質、調整閾值

### 3. 結果驗證
- **視覺化檢查**：查看生成的可視化結果
- **統計分析**：對比不同模型的指標
- **錯誤樣本分析**：找出模型失敗的案例

## 最佳實踐

1. **測試前準備**：確保測試集與訓練集完全分離
2. **多次測試**：進行多次測試確保結果穩定性
3. **閾值調優**：根據應用需求調整信心度閾值
4. **類別平衡**：注意各類別樣本數量的影響
5. **結果記錄**：詳細記錄每次測試的設定和結果
6. **視覺驗證**：結合數值指標和視覺檢查進行綜合評估
