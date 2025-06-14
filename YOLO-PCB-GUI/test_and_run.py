#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO-PCB GUI 測試與啟動腳本
執行基本功能測試並啟動GUI應用程式

作者: AI Assistant
版本: 1.1.0
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gui_test.log')
    ]
)
logger = logging.getLogger(__name__)

def check_python_version():
    """檢查Python版本"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        logger.error(f"需要Python 3.7或更高版本，當前版本: {version.major}.{version.minor}")
        return False
    logger.info(f"Python版本檢查通過: {version.major}.{version.minor}")
    return True

def check_dependencies():
    """檢查基本依賴"""
    required_modules = [
        'PyQt5',
        'numpy',
        'opencv-python', 
        'torch',
        'torchvision',
        'PIL',
        'yaml',
        'matplotlib'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            if module == 'opencv-python':
                import cv2
            elif module == 'PIL':
                import PIL
            elif module == 'yaml':
                import yaml
            else:
                __import__(module)
            logger.info(f"✓ {module}")
        except ImportError:
            missing_modules.append(module)
            logger.warning(f"✗ {module}")
    
    if missing_modules:
        logger.error(f"缺少依賴: {', '.join(missing_modules)}")
        logger.info("請執行以下命令安裝依賴:")
        logger.info("pip install -r requirements_gui.txt")
        return False
    
    logger.info("所有依賴檢查通過")
    return True

def test_gui_components():
    """測試GUI組件"""
    logger.info("開始GUI組件測試...")
    
    try:
        # 測試PyQt5基本功能
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import QCoreApplication
        
        # 創建臨時應用實例（如果不存在）
        if not QCoreApplication.instance():
            app = QApplication(sys.argv)
            test_app = True
        else:
            test_app = False
        
        # 測試導入主要模組
        from utils.config_manager import ConfigManager
        logger.info("✓ 配置管理器")
        
        from gui.widgets.image_viewer import AnnotatedImageViewer
        logger.info("✓ 圖像顯示組件")
        
        from gui.main_window import MainWindow
        logger.info("✓ 主視窗")
        
        from gui.detect_tab import DetectTab
        logger.info("✓ 檢測頁籤")
        
        from gui.train_tab import TrainTab
        logger.info("✓ 訓練頁籤")
        
        from gui.test_tab import TestTab
        logger.info("✓ 測試頁籤")
        
        # 清理臨時應用
        if test_app:
            app.quit()
        
        logger.info("GUI組件測試通過")
        return True
        
    except Exception as e:
        logger.error(f"GUI組件測試失敗: {str(e)}")
        return False

def test_core_modules():
    """測試核心模組"""
    logger.info("開始核心模組測試...")
    
    try:
        from core.detector import DetectionWorker
        logger.info("✓ 檢測核心")
        
        from core.trainer import TrainingWorker
        logger.info("✓ 訓練核心")
        
        from core.tester import TestingWorker
        logger.info("✓ 測試核心")
        
        logger.info("核心模組測試通過")
        return True
        
    except Exception as e:
        logger.error(f"核心模組測試失敗: {str(e)}")
        logger.warning("將使用模擬模式運行")
        return False

def create_sample_config():
    """創建範例配置"""
    try:
        from utils.config_manager import ConfigManager
        
        config_manager = ConfigManager()
        
        # 檢查是否已有配置
        if config_manager.config_file.exists():
            logger.info("配置檔案已存在")
            return True
        
        # 創建預設配置
        default_config = {
            'app': {
                'name': 'YOLO-PCB GUI',
                'version': '1.0.0',
                'window_geometry': [100, 100, 1200, 800],
                'theme': 'default'
            },
            'detection': {
                'default_weights': './weights/best.pt',
                'default_conf': 0.25,
                'default_iou': 0.45,
                'default_device': 'auto'
            },
            'training': {
                'default_epochs': 100,
                'default_batch_size': 16,
                'default_lr': 0.01,
                'default_patience': 50
            },
            'testing': {
                'default_conf': 0.001,
                'default_iou': 0.6,
                'default_batch_size': 32
            },
            'paths': {
                'weights_dir': './weights',
                'data_dir': './data',
                'runs_dir': './runs'
            }
        }
        
        # 保存配置
        for section, values in default_config.items():
            for key, value in values.items():
                config_manager.set(f'{section}.{key}', value)
        
        config_manager.save()
        logger.info("已創建預設配置檔案")
        return True
        
    except Exception as e:
        logger.error(f"創建配置失敗: {str(e)}")
        return False

def run_gui():
    """啟動GUI應用程式"""
    logger.info("啟動YOLO-PCB GUI...")
    
    try:
        # 導入主程式
        from main import main
        
        # 啟動應用程式
        main()
        
    except KeyboardInterrupt:
        logger.info("用戶中斷程序")
    except Exception as e:
        logger.error(f"啟動GUI失敗: {str(e)}")
        return False
    
    return True

def main():
    """主函數"""
    print("="*60)
    print("YOLO-PCB GUI 測試與啟動腳本")
    print("="*60)
    
    # 切換到腳本目錄
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    logger.info(f"工作目錄: {script_dir}")
    
    # 執行檢查
    if not check_python_version():
        return False
    
    if not check_dependencies():
        return False
    
    # 創建範例配置
    create_sample_config()
    
    # 測試組件
    gui_ok = test_gui_components()
    core_ok = test_core_modules()
    
    if not gui_ok:
        logger.error("GUI組件測試失敗，無法啟動應用程式")
        return False
    
    if not core_ok:
        logger.warning("核心模組測試失敗，將以模擬模式運行")
    
    print("="*60)
    print("所有檢查完成，啟動GUI應用程式...")
    print("="*60)
    
    # 啟動GUI
    return run_gui()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
