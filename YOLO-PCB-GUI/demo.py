#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO-PCB GUI 演示範例
展示各種功能的使用方法

作者: AI Assistant
版本: 1.0.0
"""

import os
import sys
import logging
from pathlib import Path

# 添加專案路徑到系統路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demo_config_manager():
    """演示配置管理器"""
    print("\n" + "="*50)
    print("配置管理器演示")
    print("="*50)
    
    try:
        from utils.config_manager import ConfigManager
        
        # 創建配置管理器
        config = ConfigManager()
        
        # 設置一些配置
        config.set('demo.test_value', 'Hello YOLO-PCB!')
        config.set('demo.number_value', 42)
        config.set('demo.list_value', [1, 2, 3, 4, 5])
        
        # 讀取配置
        test_value = config.get('demo.test_value', 'default')
        number_value = config.get('demo.number_value', 0)
        list_value = config.get('demo.list_value', [])
        
        print(f"字串值: {test_value}")
        print(f"數字值: {number_value}")
        print(f"列表值: {list_value}")
        
        # 保存配置
        config.save()
        print("配置已保存到檔案")
        
        return True
        
    except Exception as e:
        logger.error(f"配置管理器演示失敗: {str(e)}")
        return False

def demo_image_viewer():
    """演示圖像顯示組件"""
    print("\n" + "="*50)
    print("圖像顯示組件演示")
    print("="*50)
    
    try:
        import numpy as np
        from PyQt5.QtWidgets import QApplication
        from gui.widgets.image_viewer import AnnotatedImageViewer
        
        # 創建應用程式實例
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        # 創建圖像查看器
        viewer = AnnotatedImageViewer()
        
        # 創建示例圖像
        height, width = 480, 640
        image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # 創建示例標註
        annotations = [
            {
                'bbox': [100, 100, 200, 200],
                'label': 'defect_1: 0.85',
                'color': (0, 255, 0)
            },
            {
                'bbox': [300, 200, 400, 350],
                'label': 'defect_2: 0.72',
                'color': (255, 0, 0)
            }
        ]
        
        # 設置圖像和標註
        viewer.set_image(image)
        viewer.set_annotations(annotations)
        
        print("圖像查看器創建成功")
        print(f"圖像尺寸: {width}x{height}")
        print(f"標註數量: {len(annotations)}")
        
        # 不顯示視窗，只測試功能
        return True
        
    except Exception as e:
        logger.error(f"圖像顯示組件演示失敗: {str(e)}")
        return False

def demo_detector():
    """演示檢測器核心"""
    print("\n" + "="*50)
    print("檢測器核心演示")
    print("="*50)
    
    try:
        from core.detector import DetectionWorker
        
        # 創建檢測工作執行緒
        detector = DetectionWorker()
        
        # 設置參數
        params = {
            'source': './samples/',  # 假設有範例圖片
            'weights': './weights/best.pt',  # 假設有權重檔案
            'conf': 0.25,
            'iou': 0.45,
            'device': 'cpu'
        }
        
        detector.set_parameters(params)
        print("檢測器參數設置成功")
        print(f"來源: {params['source']}")
        print(f"權重: {params['weights']}")
        
        return True
        
    except Exception as e:
        logger.error(f"檢測器核心演示失敗: {str(e)}")
        print("將使用模擬檢測模式")
        return False

def create_demo_data():
    """創建演示資料"""
    print("\n" + "="*50)
    print("創建演示資料")
    print("="*50)
    
    try:
        # 創建必要的目錄
        dirs_to_create = [
            'weights',
            'data', 
            'runs/detect',
            'runs/train',
            'runs/test',
            'samples'
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"✓ 創建目錄: {dir_path}")
        
        # 創建範例配置檔案
        sample_yaml = """
# YOLO-PCB 資料配置範例
train: ./data/train/images
val: ./data/val/images
test: ./data/test/images

nc: 6  # 類別數量
names: ['missinghole', 'mousebite', 'opencircuit', 'short', 'spur', 'spurious']

# 類別描述
class_descriptions:
  0: "缺失孔洞"
  1: "鼠咬"
  2: "開路"
  3: "短路"
  4: "毛刺"
  5: "雜散"
"""
        
        with open('data/pcb_demo.yaml', 'w', encoding='utf-8') as f:
            f.write(sample_yaml)
        print("✓ 創建範例配置: data/pcb_demo.yaml")
        
        # 創建 README 檔案
        readme_content = """# YOLO-PCB GUI 演示資料

這個目錄包含 YOLO-PCB GUI 應用程式的演示資料。

## 目錄結構

- `weights/`: 模型權重檔案
- `data/`: 訓練和測試資料
- `runs/`: 執行結果輸出
- `samples/`: 範例圖片

## 使用說明

1. 將你的 YOLO 權重檔案放到 `weights/` 目錄
2. 將資料配置檔案放到 `data/` 目錄  
3. 將測試圖片放到 `samples/` 目錄
4. 啟動 GUI 應用程式進行檢測、訓練或測試
"""
        
        with open('README_DEMO.md', 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print("✓ 創建說明檔案: README_DEMO.md")
        
        return True
        
    except Exception as e:
        logger.error(f"創建演示資料失敗: {str(e)}")
        return False

def run_demo():
    """執行演示"""
    print("YOLO-PCB GUI 演示程式")
    print("這個程式將演示各個組件的基本功能")
    
    # 切換到專案目錄
    os.chdir(project_root)
    
    # 執行各項演示
    demos = [
        ("配置管理器", demo_config_manager),
        ("圖像顯示組件", demo_image_viewer),
        ("檢測器核心", demo_detector),
        ("創建演示資料", create_demo_data)
    ]
    
    results = []
    for name, demo_func in demos:
        try:
            result = demo_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"{name}演示出現異常: {str(e)}")
            results.append((name, False))
    
    # 顯示結果摘要
    print("\n" + "="*50)
    print("演示結果摘要")
    print("="*50)
    
    for name, success in results:
        status = "✓ 成功" if success else "✗ 失敗"
        print(f"{name:<20} {status}")
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\n演示完成: {successful}/{total} 個組件運行正常")
    
    if successful == total:
        print("\n所有組件演示成功！可以啟動完整的GUI應用程式。")
        print("執行命令: python main.py 或 python run_gui.py")
    else:
        print("\n部分組件演示失敗，請檢查依賴安裝或錯誤日誌。")
    
    return successful == total

def main():
    """主函數"""
    try:
        return run_demo()
    except KeyboardInterrupt:
        print("\n用戶中斷演示")
        return False
    except Exception as e:
        logger.error(f"演示程式出現異常: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
