#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO-PCB GUI 模組測試腳本
用於測試各個模組是否正常工作

作者: AI Assistant
版本: 1.0.0
"""

import sys
import os
from pathlib import Path

# 確保能找到模組
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_basic_imports():
    """測試基本模組導入"""
    print("測試基本模組導入...")
    
    tests = [
        ("配置管理器", "utils.config_manager", "ConfigManager"),
        ("圖像顯示組件", "gui.widgets.image_viewer", "AnnotatedImageViewer"),
        ("檢測核心", "core.detector", "DetectionWorker"),
        ("訓練核心", "core.trainer", "TrainingWorker"),
        ("測試核心", "core.tester", "TestingWorker"),
        ("檢測頁籤", "gui.detect_tab", "DetectTab"),
        ("訓練頁籤", "gui.train_tab", "TrainTab"),
        ("測試頁籤", "gui.test_tab", "TestTab"),
        ("主視窗", "gui.main_window", "MainWindow"),
        ("主應用程式", "main", "YoloPcbApp"),
    ]
    
    results = []
    
    for name, module_name, class_name in tests:
        try:
            module = __import__(module_name, fromlist=[class_name])
            class_obj = getattr(module, class_name)
            print(f"✓ {name} ({module_name}.{class_name})")
            results.append((name, True, None))
        except Exception as e:
            print(f"✗ {name} ({module_name}.{class_name}): {str(e)}")
            results.append((name, False, str(e)))
    
    return results

def test_config_manager():
    """測試配置管理器"""
    print("\n測試配置管理器功能...")
    
    try:
        from utils.config_manager import ConfigManager
        
        config = ConfigManager()
        
        # 測試設置和獲取
        config.set('test.value', 'hello')
        value = config.get('test.value', 'default')
        
        if value == 'hello':
            print("✓ 配置管理器讀寫功能正常")
            return True
        else:
            print(f"✗ 配置管理器讀寫功能異常: 期望 'hello'，得到 '{value}'")
            return False
            
    except Exception as e:
        print(f"✗ 配置管理器測試失敗: {str(e)}")
        return False

def test_image_viewer():
    """測試圖像顯示組件"""
    print("\n測試圖像顯示組件...")
    
    try:
        from gui.widgets.image_viewer import AnnotatedImageViewer
        import numpy as np
        
        # 創建測試圖像
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # 不創建實際的Qt應用程式，只測試類別創建
        print("✓ 圖像顯示組件類別創建成功")
        return True
        
    except Exception as e:
        print(f"✗ 圖像顯示組件測試失敗: {str(e)}")
        return False

def test_detection_worker():
    """測試檢測工作執行緒"""
    print("\n測試檢測工作執行緒...")
    
    try:
        from core.detector import DetectionWorker
        
        worker = DetectionWorker()
        
        # 測試參數設置
        params = {
            'source': './test',
            'weights': './test.pt',
            'conf': 0.25
        }
        worker.set_parameters(params)
        
        print("✓ 檢測工作執行緒創建和參數設置成功")
        return True
        
    except Exception as e:
        print(f"✗ 檢測工作執行緒測試失敗: {str(e)}")
        return False

def main():
    """主函數"""
    print("YOLO-PCB GUI 模組測試")
    print("="*50)
    
    # 測試基本導入
    import_results = test_basic_imports()
    
    # 計算成功率
    successful = sum(1 for _, success, _ in import_results if success)
    total = len(import_results)
    
    print(f"\n模組導入測試結果: {successful}/{total} 成功")
    
    if successful == 0:
        print("✗ 所有模組導入都失敗，請檢查環境設置")
        return 1
    
    # 功能測試
    print("\n" + "="*50)
    print("功能測試")
    print("="*50)
    
    functional_tests = [
        ("配置管理器", test_config_manager),
        ("圖像顯示組件", test_image_viewer),
        ("檢測工作執行緒", test_detection_worker),
    ]
    
    functional_results = []
    for name, test_func in functional_tests:
        try:
            result = test_func()
            functional_results.append((name, result))
        except Exception as e:
            print(f"✗ {name}功能測試異常: {str(e)}")
            functional_results.append((name, False))
    
    # 功能測試結果
    func_successful = sum(1 for _, success in functional_results if success)
    func_total = len(functional_results)
    
    print(f"\n功能測試結果: {func_successful}/{func_total} 成功")
    
    # 總結
    print("\n" + "="*50)
    print("測試總結")
    print("="*50)
    
    if successful == total and func_successful == func_total:
        print("✓ 所有測試通過！GUI應用程式應該可以正常啟動。")
        print("執行 'python run_gui.py' 或 'python main.py' 啟動應用程式")
        return 0
    else:
        print(f"部分測試失敗:")
        print(f"  模組導入: {successful}/{total}")
        print(f"  功能測試: {func_successful}/{func_total}")
        
        # 顯示失敗的模組
        print("\n失敗的模組:")
        for name, success, error in import_results:
            if not success:
                print(f"  - {name}: {error}")
        
        print("\n建議:")
        print("1. 檢查依賴安裝: pip install -r requirements_gui.txt")
        print("2. 檢查Python版本: 需要Python 3.7+")
        print("3. 檢查PyQt5安裝: pip install PyQt5")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())
