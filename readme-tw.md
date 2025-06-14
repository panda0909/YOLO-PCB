## 使用 YOLOv5 進行 PCB 缺陷檢測
我們的期刊論文「基於深度上下文學習的 PCB 缺陷檢測模型與異常趨勢警報系統」已被 Results in Engineering (RINENG) 期刊接受，可在[此處](https://doi.org/10.1016/j.rineng.2023.100968)找到。

## 範例推論結果
<img src="/samples/missinghole_1.jpg" width="200"> <img src="/samples/missinghole_2.jpg" width="200"> <img src="/samples/missinghole_3.jpg" width="200"> <img src="/samples/missinghole_4.jpg" width="200"> <img src="/samples/mousebite_1.jpg" width="200"> <img src="/samples/mousebite_2.jpg" width="200"> <img src="/samples/mousebite_3.jpg" width="200"> <img src="/samples/mousebite_4.jpg" width="200"> <img src="/samples/opencircuit_1.jpg" width="200"> <img src="/samples/opencircuit_2.jpg" width="200"> <img src="/samples/opencircuit_3.jpg" width="200"> <img src="/samples/opencircuit_4.jpg" width="200"> <img src="/samples/short_1.jpg" width="200"> <img src="/samples/short_2.jpg" width="200"> <img src="/samples/short_3.jpg" width="200"> <img src="/samples/short_4.jpg" width="200"> <img src="/samples/spur_1.jpg" width="200"> <img src="/samples/spur_2.jpg" width="200"> <img src="/samples/spur_3.jpg" width="200"> <img src="/samples/spur_4.jpg" width="200"> <img src="/samples/spurious_1.jpg" width="200"> <img src="/samples/spurious_2.jpg" width="200"> <img src="/samples/spurious_3.jpg" width="200"> <img src="/samples/spurious_4.jpg" width="200">

### 資料集詳細資訊：
1) 資料集包含 10,668 張裸露 PCB 影像，包含 6 種常見缺陷：缺孔、鼠咬、斷路、短路、毛刺和偽銅。
2) 所有影像都從 600x600 縮放至 608x608 用於訓練和測試目的。

## 開始使用
查看 YOLOv5 官方 GitHub 頁面的[教學](https://github.com/ultralytics/yolov5/blob/develop/tutorial.ipynb)以獲取更多資訊。

- 複製此存放庫
~~~
git clone https://github.com/JiaLim98/YOLO-PCB.git
~~~

- 使用 [Anaconda3](https://www.anaconda.com/download/) 準備 Python 環境。
- 建議使用 Python 3.8-3.11 版本以確保兼容性。

### 環境安裝方式一：使用 Conda（推薦）
~~~
# 創建新的 conda 環境
conda create -n yolo-pcb python=3.9
conda activate yolo-pcb

# 安裝 PyTorch（根據您的 CUDA 版本選擇）
# 對於 CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# 對於 CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# 對於 CPU 版本
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# 安裝其他依賴項
pip install -r requirements.txt
~~~

### 環境安裝方式二：使用 pip (建議使用)
~~~
# 創建虛擬環境
python -m venv yolo-pcb-env
# Windows
yolo-pcb-env\Scripts\activate
# Linux/Mac
source yolo-pcb-env/bin/activate

# 安裝 PyTorch（訪問 https://pytorch.org 獲取最新安裝命令）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安裝其他依賴項
pip install -r requirements.txt
~~~

### 兼容性說明
原始專案使用 PyTorch 1.7.1，但現代系統建議使用更新版本：
- **Python**: 3.8-3.11（避免使用 3.12+ 以確保兼容性）
- **PyTorch**: 1.12+ 或最新穩定版本
- **CUDA**: 根據您的 GPU 選擇對應版本

## 資料集準備
所有資料集必須放在 `yolov5` 資料夾旁邊。對於 YOLO 資料集格式，請按以下方式組織：
~~~
yolov5/
PCB_dataset/
  - images/
    - trainPCB/
      - *.jpg
    - valPCB/
      - *.jpg
    - testPCB/
      - *.jpg
  - labels/
    - trainPCB/
      - *.txt
    - valPCB/
      - *.txt
    - testPCB/
      - *.txt
~~~

## 訓練和測試
- 使用 YOLOv5 預訓練權重進行訓練。注意：RTX3070 在 8GB VRAM 下最大批次大小為 34
~~~
python train.py --img 608 --batch 34 --epochs 3 --data PCB.yaml --weights yolov5s.pt --nosave --cache
~~~
- 使用最終權重進行測試。注意：YOLOv5 預設使用批次大小 32
~~~
python test.py --weights ./weights/baseline_fpn_loss.pt --data PCB.yaml --img 608 --task test --batch 1
~~~

## 推論
1) 將所需影像放在 `data` 資料夾中的任何子資料夾名稱下。例如：
~~~
data/
  - PCB/
    - *.jpg
~~~
2) 使用最終權重在所需資料資料夾上執行模型推論。
~~~
python detect.py --source ./data/PCB/ --weights ./weights/baseline_fpn_loss.pt --conf 0.9
~~~

## 異常趨勢警報系統
特色的異常趨勢檢測演算法可在[此處](https://github.com/JiaLim98/YOLO-PCB/blob/main/utils/alarm.py)找到。要與您自己的模型配對，只需在模型推論或檢測期間匯入這些函數。
- [`size`](https://github.com/JiaLim98/YOLO-PCB/blob/main/utils/alarm.py#L5-L60) 檢測遞增的檢測大小（即缺陷越來越大）
- [`rep`](https://github.com/JiaLim98/YOLO-PCB/blob/main/utils/alarm.py#L62-L127) 檢測重複的局部缺陷發生（即缺陷持續在類似點累積）
- [`cnt`](https://github.com/JiaLim98/YOLO-PCB/blob/main/utils/alarm.py#L129-L165) 檢測遞增的缺陷發生（即類似缺陷重複發生，無論其位置如何）

## 致謝
YOLOv5 的所有榮譽歸於 [Ultralytics](https://github.com/ultralytics) 公司。官方 YOLOv5 GitHub 存放庫可在[此處](https://github.com/ultralytics/yolov5)找到。

資料集的所有榮譽歸於 RunWei Ding、LinHui Dai、GuangPeng Li 和 Hong Liu，他們是「TDD-net: a tiny defect detection network for printed circuit boards」的作者。他們的存放庫可在[此處](https://github.com/Ixiaohuihuihui/Tiny-Defect-Detection-for-PCB)找到。

## 專案結構說明

### 主要檔案
- `detect.py` - 模型推論腳本
- `train.py` - 模型訓練腳本
- `test.py` - 模型測試腳本
- `requirements.txt` - Python 依賴項清單

### 資料夾結構
- `data/` - 包含資料集配置檔案（PCB.yaml）和範例影像
- `models/` - YOLOv5 模型架構定義檔案
- `samples/` - 各種 PCB 缺陷的範例檢測結果影像
- `utils/` - 工具函數，包括異常趨勢警報系統
- `weights/` - 預訓練和微調後的模型權重檔案

### 支援的 PCB 缺陷類型
1. **缺孔 (missing_hole)** - PCB 上缺少應有的孔洞
2. **鼠咬 (mouse_bite)** - 類似老鼠咬痕的缺陷
3. **斷路 (open_circuit)** - 電路斷開
4. **短路 (short)** - 電路短路
5. **毛刺 (spur)** - 銅箔上的突起
6. **偽銅 (spurious_copper)** - 不應存在的銅箔

### 系統需求
- **Python**: 3.8-3.11（建議使用 3.9 或 3.10）
- **PyTorch**: 1.12+ 或最新穩定版本
- **CUDA**: 11.8 或 12.1（如果使用 GPU）
- **記憶體**: 至少 8GB RAM
- **GPU**: 建議使用 RTX3070 或更高等級的 GPU 進行訓練

### 疑難排解
如果遇到 PyTorch 安裝問題：
1. 確認 Python 版本在 3.8-3.11 之間
2. 訪問 [PyTorch 官網](https://pytorch.org) 獲取最新安裝命令
3. 根據您的 CUDA 版本選擇對應的 PyTorch 版本
4. 如果沒有 GPU，使用 CPU 版本的 PyTorch
