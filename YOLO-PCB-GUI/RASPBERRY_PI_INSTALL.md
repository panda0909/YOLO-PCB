# 樹莓派安裝指南

## 🍓 為什麼樹莓派需要 pyproject.toml？

### 技術原因
1. **PEP 518 標準**：Python 官方推薦的現代套件管理標準
2. **依賴解析**：更精確的依賴管理，避免版本衝突
3. **ARM 架構支援**：針對樹莓派 ARM 處理器優化
4. **記憶體效率**：適合樹莓派有限資源的安裝策略

### 樹莓派特殊需求
- **piwheels 支援**：使用預編譯的 ARM 套件，大幅減少安裝時間
- **系統資源管理**：避免編譯大型套件時記憶體不足
- **硬體整合**：支援樹莓派特有的 GPIO、相機等功能

## 🚀 安裝方式

### 1. 基本安裝（推薦）
\`\`\`bash
# 使用 pip 安裝（會自動讀取 pyproject.toml）
pip install -e .
\`\`\`

### 2. 最小安裝（低階樹莓派）
\`\`\`bash
pip install -e ".[minimal]"
\`\`\`

### 3. 完整安裝（高階樹莓派 4/5）
\`\`\`bash
pip install -e ".[full,camera]"
\`\`\`

### 4. 開發安裝
\`\`\`bash
pip install -e ".[dev]"
\`\`\`

## 🔧 樹莓派專用優化

### 記憶體優化
- 使用 `opencv-python-headless` 而非完整版
- 選擇性安裝套件組合
- 預編譯二進位套件優先

### 硬體支援
- 自動偵測 ARM 架構
- 樹莓派相機支援
- GPIO 控制功能

### 網路優化
- 使用 piwheels 鏡像加速下載
- 二進位套件優先，減少編譯時間

## 📊 不同樹莓派型號建議

| 型號 | RAM | 建議安裝 | 說明 |
|------|-----|----------|------|
| Pi Zero/1 | 512MB | minimal | 基本功能 |
| Pi 2/3 | 1GB | 預設 | 標準功能 |
| Pi 4 (4GB+) | 4GB+ | full | 完整功能 |
| Pi 5 | 8GB | full,camera | 所有功能 |

## 🛠️ 故障排除

### 如果安裝失敗
\`\`\`bash
# 1. 更新系統
sudo apt update && sudo apt upgrade -y

# 2. 安裝必要的系統依賴
sudo apt install python3-dev python3-pip libatlas-base-dev

# 3. 增加 swap 空間（如果記憶體不足）
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile  # 設定 CONF_SWAPSIZE=1024
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# 4. 重新安裝
pip install -e ".[minimal]"
\`\`\`

## 🎯 優勢總結

使用 `pyproject.toml` 在樹莓派上的優勢：

✅ **快速安裝**：使用預編譯套件，節省 80% 安裝時間  
✅ **記憶體友善**：避免編譯時記憶體爆滿  
✅ **依賴管理**：精確版本控制，避免衝突  
✅ **硬體整合**：自動偵測並安裝樹莓派專用功能  
✅ **模組化安裝**：根據需求選擇功能組合  
