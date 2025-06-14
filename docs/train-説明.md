# train.py 模型訓練腳本說明文件

## 概述
`train.py` 是 YOLO-PCB 專案中的模型訓練腳本，用於訓練 YOLOv5 模型進行 PCB 缺陷檢測。支援從預訓練模型開始訓練或從頭開始訓練。

## 主要功能

### 1. 訓練流程架構

#### 1.1 初始化階段
- **環境設置**：設置隨機種子、日誌記錄、分散式訓練
- **資料準備**：載入資料集配置、驗證資料集完整性
- **模型建構**：載入預訓練模型或建立新模型
- **優化器設定**：配置 SGD/Adam 優化器和學習率調度器

#### 1.2 訓練循環
- **數據載入**：批次載入訓練影像和標籤
- **前向傳播**：計算模型預測結果
- **損失計算**：計算邊界框、物件性和分類損失
- **反向傳播**：更新模型權重
- **驗證評估**：定期在驗證集上評估模型效能

#### 1.3 模型管理
- **權重保存**：保存最佳模型和最新模型
- **Early Stopping**：基於驗證結果的早期停止機制
- **模型演化**：支援超參數演化優化

### 2. 核心訓練機制

#### 2.1 混合精度訓練
```python
# 使用 PyTorch 的自動混合精度
with amp.autocast():
    pred = model(imgs)
    loss, loss_items = compute_loss(pred, targets.to(device), model)
```

#### 2.2 梯度累積
```python
# 小批次梯度累積以模擬大批次訓練
loss.backward()
if ni % accumulate == 0:
    optimizer.step()
    optimizer.zero_grad()
```

#### 2.3 指數移動平均 (EMA)
```python
# 使用 EMA 提高模型穩定性
if ema:
    ema.update(model)
```

### 3. 損失函數組成

#### 3.1 邊界框回歸損失 (Box Loss)
- 使用 IoU-based 損失（GIoU/DIoU/CIoU）
- 負責預測邊界框的準確位置和大小

#### 3.2 物件性損失 (Objectness Loss)
- 二元交叉熵損失
- 判斷網格單元是否包含物件

#### 3.3 分類損失 (Classification Loss)
- 多類別交叉熵損失
- 預測物件的具體類別（6種PCB缺陷）

### 4. 數據增強策略

#### 4.1 基礎增強
- **隨機縮放**：0.5-1.5倍縮放
- **隨機翻轉**：水平翻轉
- **顏色抖動**：HSV 色彩空間調整
- **隨機裁切**：Mosaic 和 MixUp 增強

#### 4.2 高級增強
```python
# Mosaic 增強：結合4張影像
# MixUp 增強：影像混合
# Copy-Paste 增強：實例貼上
```

## 命令列參數

### 資料和模型參數
- `--data`：資料集配置檔案（如：PCB.yaml）
- `--cfg`：模型配置檔案（如：yolov5s.yaml）
- `--weights`：預訓練權重（如：yolov5s.pt）
- `--name`：實驗名稱
- `--project`：專案保存目錄

### 訓練參數
- `--epochs`：訓練輪數（預設：300）
- `--batch-size`：批次大小（預設：16）
- `--img-size`：訓練影像尺寸（預設：640）
- `--resume`：從最後一個檢查點恢復訓練
- `--nosave`：不保存中間結果（節省空間）

### 優化參數
- `--adam`：使用 Adam 優化器（預設：SGD）
- `--sync-bn`：使用同步批次正規化
- `--workers`：數據載入工作執行緒數
- `--device`：訓練裝置（如：0,1,2,3 或 cpu）

### 增強和正規化
- `--hyp`：超參數配置檔案
- `--augment`：啟用測試時增強
- `--cache`：快取影像到記憶體/磁碟
- `--image-weights`：使用加權影像選擇
- `--multi-scale`：多尺度訓練

## 使用範例

### 1. 基礎訓練
```bash
python train.py --data PCB.yaml --weights yolov5s.pt --epochs 100 --batch-size 16
```

### 2. 高效能訓練（RTX3070）
```bash
python train.py --data PCB.yaml --weights yolov5s.pt --epochs 300 --batch-size 34 --img-size 608 --cache --device 0
```

### 3. 從頭開始訓練
```bash
python train.py --data PCB.yaml --cfg yolov5s.yaml --epochs 500 --batch-size 16
```

### 4. 恢復中斷的訓練
```bash
python train.py --resume runs/train/exp/weights/last.pt
```

### 5. 多GPU訓練
```bash
python -m torch.distributed.launch --nproc_per_node 2 train.py --data PCB.yaml --weights yolov5s.pt --device 0,1
```

## 超參數配置

### 重要超參數說明
```yaml
# 學習率相關
lr0: 0.01          # 初始學習率
lrf: 0.2           # 最終學習率（lr0 * lrf）
momentum: 0.937    # SGD 動量
weight_decay: 0.0005  # 權重衰減

# 損失函數權重
box: 0.05          # 邊界框損失權重
cls: 0.5           # 分類損失權重
cls_pw: 1.0        # 分類正樣本權重
obj: 1.0           # 物件性損失權重
obj_pw: 1.0        # 物件性正樣本權重

# 數據增強
hsv_h: 0.015       # HSV-Hue 增強
hsv_s: 0.7         # HSV-Saturation 增強
hsv_v: 0.4         # HSV-Value 增強
degrees: 0.0       # 影像旋轉角度
translate: 0.1     # 影像平移比例
scale: 0.5         # 影像縮放增益
shear: 0.0         # 影像剪切角度
perspective: 0.0   # 透視變換概率
flipud: 0.0        # 垂直翻轉概率
fliplr: 0.5        # 水平翻轉概率
mosaic: 1.0        # Mosaic 增強概率
mixup: 0.0         # MixUp 增強概率
```

## 訓練監控

### 1. TensorBoard 日誌
```bash
# 啟動 TensorBoard
tensorboard --logdir runs/train
```

### 2. Weights & Biases 整合
```python
# 自動記錄訓練指標
import wandb
wandb.init(project='YOLO-PCB')
```

### 3. 訓練指標
- **mAP@0.5**：IoU=0.5 時的平均精度
- **mAP@0.5:0.95**：IoU 0.5-0.95 的平均精度
- **Precision**：精確率
- **Recall**：召回率
- **損失值**：box_loss, obj_loss, cls_loss

## 輸出檔案結構
```
runs/train/exp/
├── weights/
│   ├── best.pt          # 最佳模型權重
│   ├── last.pt          # 最新模型權重
│   └── epoch_*.pt       # 定期檢查點
├── results.png          # 訓練曲線圖
├── confusion_matrix.png # 混淆矩陣
├── labels.jpg          # 標籤分佈圖
├── train_batch*.jpg    # 訓練批次範例
├── val_batch*.jpg      # 驗證批次範例
├── hyp.yaml           # 使用的超參數
├── opt.yaml           # 訓練選項
└── results.txt        # 詳細結果日誌
```

## 效能調優建議

### 1. 硬體優化
- **GPU 記憶體**：根據 GPU 記憶體調整批次大小
- **CPU 核心**：適當設置 `--workers` 參數
- **混合精度**：自動啟用以加速訓練

### 2. 超參數調優
- **學習率**：從較小值開始（0.01），觀察收斂情況
- **批次大小**：越大越好，受限於 GPU 記憶體
- **影像尺寸**：608 在精度和速度間取得平衡

### 3. 數據策略
- **數據增強**：根據資料集特性調整增強強度
- **類別平衡**：使用 `--image-weights` 處理類別不平衡
- **快取策略**：`--cache ram` 或 `--cache disk` 加速載入

## 常見問題與解決方案

### 1. 記憶體不足
```bash
# 減少批次大小
--batch-size 8

# 使用較小的影像尺寸
--img-size 416

# 關閉快取
# 不使用 --cache 參數
```

### 2. 訓練不收斂
```bash
# 降低學習率
--hyp hyp.finetune.yaml  # 使用微調超參數

# 增加 warmup 期
# 編輯超參數檔案中的 warmup_epochs
```

### 3. 過擬合問題
```bash
# 增加數據增強
--augment

# 使用更強的正規化
# 編輯超參數檔案增加 weight_decay
```

### 4. 驗證精度不穩定
```bash
# 使用 EMA（預設啟用）
# 增加驗證頻率
# 使用更大的驗證集
```

## 最佳實踐

1. **預訓練權重**：總是從 COCO 預訓練權重開始
2. **數據品質**：確保標註準確性，移除低品質影像
3. **類別平衡**：注意各類別樣本數量分佈
4. **驗證策略**：使用獨立的測試集進行最終評估
5. **實驗記錄**：詳細記錄每次實驗的設定和結果
6. **模型選擇**：根據精度和速度需求選擇合適的模型尺寸
