# detect.py 模型推論腳本說明文件

## 概述
`detect.py` 是 YOLO-PCB 專案中的模型推論腳本，用於對單張影像、批次影像、視頻或即時攝影機串流進行 PCB 缺陷檢測。

## 主要功能

### 1. 輸入源支援
- **影像檔案**：支援 JPG、PNG 等常見格式
- **影像資料夾**：批次處理整個資料夾中的影像
- **視頻檔案**：支援 MP4、AVI 等視頻格式
- **網路攝影機**：即時檢測（使用攝影機編號，如 0）
- **網路串流**：支援 RTSP、RTMP、HTTP 串流

### 2. 核心檢測流程

#### 2.1 模型載入與初始化
```python
# 載入預訓練模型
model = attempt_load(weights, map_location=device)
# 檢查影像尺寸兼容性
imgsz = check_img_size(imgsz, s=model.stride.max())
# 啟用半精度推論（GPU加速）
if half:
    model.half()
```

#### 2.2 影像前處理
- 將影像調整為模型要求的尺寸（預設 640x640）
- 數值正規化：0-255 → 0.0-1.0
- 數據類型轉換：uint8 → fp16/fp32
- 增加批次維度

#### 2.3 模型推論
```python
# 執行前向傳播
pred = model(img, augment=opt.augment)[0]
# 應用非極大值抑制 (NMS)
pred = non_max_suppression(pred, conf_thres, iou_thres, 
                          classes=classes, agnostic=agnostic_nms)
```

#### 2.4 後處理與結果輸出
- 將邊界框座標從模型尺寸縮放回原始影像尺寸
- 繪製檢測框和標籤
- 計算並顯示每個類別的檢測數量
- 保存結果（影像/視頻/文字檔案）

### 3. 輸出格式

#### 3.1 視覺化輸出
- 在原始影像上繪製彩色邊界框
- 顯示類別名稱和信心分數
- 支援即時顯示或保存為檔案

#### 3.2 文字格式輸出 (YOLO 格式)
```
class_id center_x center_y width height [confidence]
```

#### 3.3 統計資訊
- 每張影像的檢測數量
- 各類別缺陷的統計
- 推論時間資訊

## 命令列參數

### 必要參數
- `--weights`：模型權重檔案路徑（如：`./weights/baseline_fpn_loss.pt`）
- `--source`：輸入源路徑（影像/資料夾/視頻/攝影機編號）

### 重要可選參數
- `--img-size`：推論影像尺寸（預設：640）
- `--conf-thres`：信心度閾值（預設：0.25）
- `--iou-thres`：NMS 的 IoU 閾值（預設：0.45）
- `--device`：運算裝置（'cpu' 或 '0,1,2,3' 指定 GPU）
- `--view-img`：即時顯示檢測結果
- `--save-txt`：保存檢測結果為文字檔案
- `--classes`：指定要檢測的類別編號
- `--augment`：啟用測試時增強（TTA）

## 使用範例

### 1. 基本使用
```bash
python detect.py --source ./data/PCB/ --weights ./weights/baseline_fpn_loss.pt
```

### 2. 高精度檢測
```bash
python detect.py --source ./data/PCB/ --weights ./weights/baseline_fpn_loss.pt --conf 0.9 --img-size 608
```

### 3. 即時攝影機檢測
```bash
python detect.py --source 0 --weights ./weights/baseline_fpn_loss.pt --view-img
```

### 4. 批次處理並保存結果
```bash
python detect.py --source ./data/test_images/ --weights ./weights/baseline_fpn_loss.pt --save-txt --save-conf
```

### 5. 指定類別檢測
```bash
python detect.py --source ./data/PCB/ --weights ./weights/baseline_fpn_loss.pt --classes 0 2 4
```

## PCB 缺陷類別對應

| 類別 ID | 缺陷名稱 | 英文名稱 | 描述 |
|---------|----------|----------|------|
| 0 | 缺孔 | missing_hole | PCB 上缺少應有的孔洞 |
| 1 | 鼠咬 | mouse_bite | 邊緣呈鋸齒狀的缺陷 |
| 2 | 斷路 | open_circuit | 電路連接中斷 |
| 3 | 短路 | short | 不應連接的電路點相連 |
| 4 | 毛刺 | spur | 銅箔上的多餘突起 |
| 5 | 偽銅 | spurious_copper | 不應存在的銅箔沉積 |

## 效能優化建議

### 1. GPU 加速
- 使用 CUDA 支援的 GPU
- 啟用半精度推論（自動啟用）
- 適當設置批次大小

### 2. 推論優化
- 根據精度需求調整 `--conf-thres`
- 使用適當的影像尺寸（608/640/800）
- 考慮使用 `--augment` 提高檢測精度

### 3. 記憶體管理
- 處理大批量影像時注意記憶體使用
- 視頻處理時考慮幀率和解析度

## 輸出檔案結構
```
runs/detect/exp/
├── labels/          # YOLO 格式的檢測結果文字檔案
│   ├── image1.txt
│   └── image2.txt
├── image1.jpg       # 帶有檢測框的結果影像
├── image2.jpg
└── ...
```

## 注意事項

1. **模型相容性**：確保使用的權重檔案與當前程式碼版本相容
2. **影像格式**：支援常見影像格式，建議使用 JPG 或 PNG
3. **記憶體需求**：大影像或批次處理需要足夠的 GPU/CPU 記憶體
4. **閾值調整**：根據實際應用場景調整信心度和 IoU 閾值
5. **類別過濾**：可使用 `--classes` 參數僅檢測特定類型的缺陷

## 錯誤排除

1. **CUDA 記憶體不足**：降低影像尺寸或使用 CPU 模式
2. **模型載入失敗**：檢查權重檔案路徑和完整性
3. **影像讀取錯誤**：確認影像檔案格式和路徑正確
4. **攝影機無法開啟**：檢查攝影機權限和驅動程式

## 網路攝影機即時檢測指南

### 攝影機設備準備

#### 1. 硬體需求
- **網路攝影機**：USB 攝影機或內建攝影機
- **照明設備**：均勻的白光照明（避免陰影和反射）
- **固定支架**：確保攝影機穩定，避免晃動
- **PCB 檢測平台**：平整的檢測台面，建議使用深色背景

#### 2. 攝影機配置
- **解析度**：建議 1280x720 或更高
- **幀率**：至少 15 FPS，推薦 30 FPS
- **自動對焦**：關閉自動對焦，手動調整到最佳清晰度
- **曝光設定**：固定曝光值，避免自動調整造成影像不穩定

### 即時檢測命令

#### 1. 基本即時檢測
```bash
# 使用預設攝影機（編號 0）
python detect.py --source 0 --weights ./weights/baseline_fpn_loss.pt --view-img --conf 0.7

# 指定特定攝影機
python detect.py --source 1 --weights ./weights/baseline_fpn_loss.pt --view-img --conf 0.7
```

#### 2. 高品質即時檢測
```bash
# 使用較大影像尺寸和高信心度閾值
python detect.py --source 0 --weights ./weights/baseline_fpn_loss.pt --img-size 608 --conf 0.8 --view-img --device 0
```

#### 3. 即時檢測並記錄結果
```bash
# 同時顯示和保存檢測結果
python detect.py --source 0 --weights ./weights/baseline_fpn_loss.pt --view-img --save-txt --save-conf --conf 0.75
```

#### 4. 生產線即時監控
```bash
# 針對生產環境的配置
python detect.py --source 0 --weights ./weights/baseline_fpn_loss.pt --img-size 640 --conf 0.85 --iou-thres 0.4 --view-img --device 0 --save-txt --project runs/production --name realtime_detection
```

### 攝影機參數優化

#### 1. 檢查可用攝影機
```python
# 在 Python 中測試攝影機
import cv2

# 測試攝影機 0 到 3
for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"攝影機 {i} 可用")
        ret, frame = cap.read()
        if ret:
            print(f"攝影機 {i} 解析度: {frame.shape}")
        cap.release()
    else:
        print(f"攝影機 {i} 不可用")
```

#### 2. 攝影機設定腳本
```python
# camera_setup.py - 攝影機參數調整腳本
import cv2

def setup_camera(camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    
    # 設定解析度
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 設定幀率
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # 關閉自動對焦
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    
    # 設定固定曝光
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 手動曝光
    cap.set(cv2.CAP_PROP_EXPOSURE, -5)  # 曝光值
    
    # 設定亮度和對比度
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
    cap.set(cv2.CAP_PROP_CONTRAST, 0.5)
    
    return cap

# 使用範例
cap = setup_camera(0)
while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Camera Setup', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
```

### 效能優化設定

#### 1. GPU 加速配置
```bash
# 確保使用 GPU 進行即時推論
ㄏ
```

#### 2. 降低延遲設定
```bash
# 使用較小的影像尺寸以提高速度
python detect.py --source 0 --weights ./weights/baseline_fpn_loss.pt --img-size 416 --view-img --conf 0.7
```

#### 3. 批次處理優化
```bash
# 不保存中間結果以提高速度
python detect.py --source 0 --weights ./weights/baseline_fpn_loss.pt --view-img --nosave
```

### 檢測結果即時處理

#### 1. 自訂檢測結果處理
```python
# 修改 detect.py 添加自訂處理邏輯
def process_detection_results(detections, frame_count):
    """處理檢測結果的自訂函數"""
    defect_count = {
        'missing_hole': 0,
        'mouse_bite': 0,
        'open_circuit': 0,
        'short': 0,
        'spur': 0,
        'spurious_copper': 0
    }
    
    total_defects = len(detections)
    
    # 統計各類別缺陷數量
    for det in detections:
        class_name = names[int(det[-1])]
        defect_count[class_name] += 1
    
    # 輸出統計結果
    if total_defects > 0:
        print(f"幀 {frame_count}: 發現 {total_defects} 個缺陷")
        for defect_type, count in defect_count.items():
            if count > 0:
                print(f"  {defect_type}: {count} 個")
    
    return defect_count
```

#### 2. 品質控制閾值
```python
# 設定品質控制標準
QUALITY_THRESHOLDS = {
    'max_defects_per_pcb': 2,      # 每片 PCB 最大允許缺陷數
    'critical_defects': ['short', 'open_circuit'],  # 嚴重缺陷類型
    'confidence_threshold': 0.8     # 嚴重缺陷的最低信心度
}

def quality_control_check(detections):
    """品質控制檢查"""
    total_defects = len(detections)
    critical_defects = []
    
    for det in detections:
        class_name = names[int(det[-1])]
        confidence = det[-2]
        
        if class_name in QUALITY_THRESHOLDS['critical_defects']:
            if confidence >= QUALITY_THRESHOLDS['confidence_threshold']:
                critical_defects.append((class_name, confidence))
    
    # 判斷品質狀態
    if len(critical_defects) > 0:
        return "REJECT", critical_defects
    elif total_defects > QUALITY_THRESHOLDS['max_defects_per_pcb']:
        return "REVIEW", []
    else:
        return "PASS", []
```

### 生產環境部署

#### 1. 工業攝影機整合
```bash
# 使用工業攝影機（通常需要特殊驅動）
# 支援 GigE Vision 或 USB3 Vision 協議
python detect.py --source "gige://192.168.1.100" --weights ./weights/baseline_fpn_loss.pt --view-img
```

#### 2. 多攝影機同時檢測
```python
# multi_camera_detect.py
import threading
import cv2
import torch

def detect_camera(camera_id, model, device):
    """單一攝影機檢測執行緒"""
    cap = cv2.VideoCapture(camera_id)
    
    while True:
        ret, frame = cap.read()
        if ret:
            # 執行檢測邏輯
            results = model(frame)
            # 處理結果...
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()

# 啟動多個檢測執行緒
threads = []
for camera_id in [0, 1, 2]:  # 三個攝影機
    thread = threading.Thread(target=detect_camera, args=(camera_id, model, device))
    thread.start()
    threads.append(thread)
```

#### 3. 結果記錄和追蹤
```bash
# 建立每日檢測記錄
python detect.py --source 0 --weights ./weights/baseline_fpn_loss.pt --save-txt --project runs/production/$(date +%Y%m%d) --name camera_0
```

### 故障排除

#### 1. 沒有畫面與反應的問題診斷

##### 問題 1：攝影機權限或占用問題
```python
# 攝影機診斷腳本 - camera_test.py
import cv2
import sys

def test_camera(camera_id=0):
    print(f"測試攝影機 {camera_id}...")
    
    # 嘗試開啟攝影機
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ 攝影機 {camera_id} 無法開啟")
        return False
    
    print(f"✅ 攝影機 {camera_id} 開啟成功")
    
    # 測試讀取畫面
    ret, frame = cap.read()
    if not ret:
        print("❌ 無法讀取攝影機畫面")
        cap.release()
        return False
    
    print(f"✅ 畫面讀取成功，解析度: {frame.shape[:2]}")
    
    # 顯示測試畫面
    cv2.imshow(f'Camera {camera_id} Test', frame)
    print("✅ 測試視窗已開啟，按任意鍵關閉...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()
    
    return True

# 測試攝影機 0 到 3
for i in range(4):
    if test_camera(i):
        print(f"✅ 攝影機 {i} 可用於檢測")
        break
else:
    print("❌ 未找到可用的攝影機")
```

##### 問題 2：檢查程序是否正常啟動
```bash
# 在命令列執行檢測時加上詳細輸出
python detect.py --source 0 --weights ./weights/baseline_fpn_loss.pt --view-img --conf 0.7 --device cpu

# 如果上述無反應，檢查 Python 環境
python --version
python -c "import cv2; print('OpenCV 版本:', cv2.__version__)"
python -c "import torch; print('PyTorch 版本:', torch.__version__)"
```

##### 問題 3：檢查權重檔案和路徑
```bash
# 檢查權重檔案是否存在
dir weights\
# 或在 Linux/Mac
ls -la weights/

# 檢查權重檔案完整性
python -c "import torch; print('權重載入測試:', torch.load('./weights/baseline_fpn_loss.pt', map_location='cpu').keys())"
```

##### 問題 4：Windows 攝影機後端問題
```python
# Windows 攝影機後端測試 - windows_camera_fix.py
import cv2

def test_windows_backends():
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_V4L2, "Video4Linux"),
        (None, "預設後端")
    ]
    
    for backend, name in backends:
        print(f"\n測試 {name}...")
        try:
            if backend is None:
                cap = cv2.VideoCapture(0)
            else:
                cap = cv2.VideoCapture(0, backend)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"✅ {name} 成功！解析度: {frame.shape}")
                    cv2.imshow(f'{name} Test', frame)
                    cv2.waitKey(1000)  # 顯示 1 秒
                    cv2.destroyAllWindows()
                    cap.release()
                    return backend
                else:
                    print(f"❌ {name} 無法讀取畫面")
            else:
                print(f"❌ {name} 無法開啟攝影機")
            cap.release()
        except Exception as e:
            print(f"❌ {name} 發生錯誤: {e}")
    
    return None

# 執行測試
working_backend = test_windows_backends()
if working_backend:
    print(f"\n建議使用的後端: {working_backend}")
```

##### 問題 5：修改 detect.py 以支援 Windows DirectShow
```python
# 在 detect.py 中找到攝影機初始化部分，修改為：
if webcam:
    view_img = True
    cudnn.benchmark = True
    # Windows 系統使用 DirectShow 後端
    import platform
    if platform.system() == 'Windows':
        dataset = LoadStreams(source, img_size=imgsz, backend=cv2.CAP_DSHOW)
    else:
        dataset = LoadStreams(source, img_size=imgsz)
```

#### 2. 攝影機無法開啟的一般解決方案
```python
# 檢查攝影機狀態
import cv2
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("錯誤：無法開啟攝影機")
    # 嘗試不同的後端
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows
    # cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Linux
```

#### Quick Fix 快速解決方案

**Step 1: 測試攝影機**
```bash
# 建立並執行攝影機測試腳本
python -c "
import cv2
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows 專用
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print('攝影機正常，解析度:', frame.shape)
        cv2.imshow('Test', frame)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
    else:
        print('攝影機無法讀取畫面')
else:
    print('攝影機無法開啟')
cap.release()
"
```

**Step 2: 修改檢測命令**
```bash
# 如果攝影機測試成功，使用以下命令
python detect.py --source 0 --weights ./weights/baseline_fpn_loss.pt --view-img --conf 0.7 --device cpu --img-size 640

# 如果仍無反應，嘗試指定攝影機後端
python detect.py --source 0 --weights ./weights/baseline_fpn_loss.pt --view-img --conf 0.7 --nosave
```

**Step 3: 環境診斷（如果上述命令沒有輸出）**

如果執行 `python --version` 等命令沒有任何輸出，請按以下步驟診斷：

```powershell
# 1. 檢查 Python 是否安裝
where python
# 如果沒有輸出，Python 可能沒有安裝或不在 PATH 中

# 2. 檢查是否有 Python 的其他版本
where python3
where py

# 3. 嘗試使用 Python Launcher（Windows）
py --version
py -c "import sys; print('Python 路徑:', sys.executable)"

# 4. 檢查 Conda 環境
conda --version
conda info --envs

# 5. 如果使用 Conda，激活環境
conda activate yolo-pcb
python --version
```

**環境激活成功後的驗證步驟：**

```powershell
# 1. 確認 Python 版本
python --version

# 2. 檢查必要套件
python -c "import cv2; print('✅ OpenCV 版本:', cv2.__version__)"
python -c "import torch; print('✅ PyTorch 版本:', torch.__version__)"
python -c "import numpy; print('✅ NumPy 版本:', numpy.__version__)"

# 3. 檢查 CUDA 支援（如果有 GPU）
python -c "import torch; print('CUDA 可用:', torch.cuda.is_available())"

# 4. 測試攝影機連接
python -c "
import cv2
print('測試攝影機...')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print('✅ 攝影機正常，解析度:', frame.shape)
    else:
        print('❌ 攝影機無法讀取畫面')
    cap.release()
else:
    print('❌ 攝影機無法開啟')
"

# 5. 如果上述都正常，測試 YOLO 檢測
python detect.py --source 0 --weights ./weights/baseline_fpn_loss.pt --view-img --conf 0.7 --device cpu
```

**如果套件缺失，請安裝：**

```powershell
# 確保在 yolo-pcb 環境中
conda activate yolo-pcb

# 安裝基本套件
pip install opencv-python
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安裝專案依賴
pip install -r requirements.txt

# 或者逐個安裝主要依賴
pip install numpy matplotlib pillow pyyaml scipy tensorboard tqdm seaborn pandas thop pycocotools
```

**選項 1：使用 Conda 安裝環境**
```powershell
# 下載並安裝 Miniconda
# https://docs.conda.io/en/latest/miniconda.html

# 創建新環境
conda create -n yolo-pcb python=3.9 -y
conda activate yolo-pcb

# 安裝 PyTorch（CPU 版本）
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# 安裝 OpenCV 和其他依賴
pip install opencv-python
pip install -r requirements.txt

# 驗證安裝
python -c "import cv2; print('OpenCV 版本:', cv2.__version__)"
python -c "import torch; print('PyTorch 版本:', torch.__version__)"
```

**選項 2：使用 Python 官方版本**
```powershell
# 1. 從 https://www.python.org/downloads/ 下載 Python 3.9
# 2. 安裝時勾選 "Add Python to PATH"
# 3. 重新開啟 PowerShell
python --version

# 安裝套件
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python
pip install -r requirements.txt
```

**選項 3：PowerShell 執行策略修復**
```powershell
# 檢查執行策略
Get-ExecutionPolicy

# 如果是 Restricted，更改為 RemoteSigned
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 重新嘗試 Python 命令
python --version
```

**Step 4: 檢查常見問題**
- ✅ Python 是否正確安裝並在 PATH 中
- ✅ 必要套件是否已安裝（torch, opencv-python）
- ✅ PowerShell 執行策略是否允許執行
- ✅ 攝影機是否被其他程式占用（關閉 Skype、Teams 等）
- ✅ 是否有攝影機使用權限
- ✅ 檢查防毒軟體是否阻擋攝影機存取
- ✅ 更新攝影機驅動程式
- ✅ 檢查 USB 連接是否穩定

#### 2. 延遲過高問題
```bash
# 減少處理負載
python detect.py --source 0 --weights yolov5s.pt --img-size 320 --conf 0.5 --view-img

# 使用更快的模型
python detect.py --source 0 --weights yolov5n.pt --view-img
```

#### 3. 檢測精度不佳
- **照明調整**：確保均勻照明，避免陰影
- **焦距調整**：手動調整攝影機焦距
- **背景優化**：使用對比度高的背景
- **穩定性**：確保攝影機固定，避免晃動

### 即時檢測最佳實踐

1. **環境準備**：
   - 控制光照條件，使用穩定的白光照明
   - 確保 PCB 與背景有足夠對比度
   - 固定攝影機位置和角度

2. **參數設定**：
   - 根據檢測精度需求調整 `--conf` 閾值
   - 使用適當的 `--img-size` 平衡速度和精度
   - 啟用 GPU 加速（`--device 0`）

3. **品質控制**：
   - 設定合理的檢測閾值
   - 建立缺陷嚴重性分級
   - 記錄檢測歷史和統計資料

4. **系統穩定性**：
   - 定期重啟檢測程序
   - 監控系統資源使用情況
   - 建立錯誤恢復機制

## PyTorch 2.6+ 權重載入問題

**問題描述：**
PyTorch 2.6+ 版本改變了 `torch.load` 的預設行為，導致載入權重時出現 `_pickle.UnpicklingError: Weights only load failed` 錯誤。

**錯誤訊息：**
```
_pickle.UnpicklingError: Weights only load failed...
WeightsUnpickler error: Unsupported global: GLOBAL models.yolo.Model was not an allowed global by default
```

**解決方案 1：修改 experimental.py（推薦）**
```python
# 修改 models/experimental.py 第 118 行
# 從：
model.append(torch.load(w, map_location=map_location)['model'].float().fuse().eval())

# 改為：
model.append(torch.load(w, map_location=map_location, weights_only=False)['model'].float().fuse().eval())
```

**解決方案 2：使用 PowerShell 快速修復**
```powershell
# 在 YOLO-PCB 目錄下執行
(Get-Content models\experimental.py) -replace "torch\.load\(w, map_location=map_location\)", "torch.load(w, map_location=map_location, weights_only=False)" | Set-Content models\experimental.py
```

**解決方案 3：建立修復腳本**
```python
# 建立 fix_pytorch_load.py 檔案
import re

def fix_pytorch_load_issue():
    file_path = 'models/experimental.py'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修正 torch.load 調用
    pattern = r'torch\.load\(([^)]+)\)'
    replacement = r'torch.load(\1, weights_only=False)'
    
    # 避免重複修改
    if 'weights_only=False' not in content:
        content = re.sub(pattern, replacement, content)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ 已修復 PyTorch 載入問題")
    else:
        print("✅ 已經修復過了")

if __name__ == "__main__":
    fix_pytorch_load_issue()
```

**執行修復：**
```powershell
# 方式 1：使用 PowerShell 一行修復
(Get-Content models\experimental.py) -replace "torch\.load\(w, map_location=map_location\)", "torch.load(w, map_location=map_location, weights_only=False)" | Set-Content models\experimental.py

# 方式 2：使用 Python 腳本修復
python -c "
import re
with open('models/experimental.py', 'r') as f: content = f.read()
if 'weights_only=False' not in content:
    content = re.sub(r'torch\.load\(w, map_location=map_location\)', 'torch.load(w, map_location=map_location, weights_only=False)', content)
    with open('models/experimental.py', 'w') as f: f.write(content)
    print('✅ 修復完成')
else:
    print('✅ 已經修復過了')
"

# 修復後重新執行檢測
python detect.py --source 0 --weights yolov5s.pt --img-size 320 --conf 0.5 --view-img
```

**驗證修復：**
```powershell
# 檢查修改是否成功
findstr "weights_only=False" models\experimental.py
```

**解決方案 4：使用專案權重檔案**
```powershell
# 如果修復後仍有問題，使用專案提供的權重
python detect.py --source 0 --weights ./weights/baseline_fpn_loss.pt --img-size 320 --conf 0.5 --view-img --device cpu
```
