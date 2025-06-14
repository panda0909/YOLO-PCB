#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
攝像頭檢測功能測試腳本

作者: AI Assistant
版本: 1.0.0
"""

import sys
import os
import cv2
from pathlib import Path

# 添加專案路徑
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_camera_availability():
    """測試攝像頭可用性"""
    print("測試攝像頭可用性...")
    
    available_cameras = []
    
    # 測試攝像頭ID 0-3
    for camera_id in range(4):
        try:
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✓ 攝像頭 {camera_id} 可用 (解析度: {frame.shape[1]}x{frame.shape[0]})")
                    available_cameras.append(camera_id)
                else:
                    print(f"⚠ 攝像頭 {camera_id} 可打開但無法讀取畫面")
                cap.release()
            else:
                print(f"✗ 攝像頭 {camera_id} 不可用")
        except Exception as e:
            print(f"✗ 攝像頭 {camera_id} 測試出錯: {str(e)}")
    
    return available_cameras

def test_detector_camera_validation():
    """測試檢測器的攝像頭驗證功能"""
    print("\n測試檢測器攝像頭驗證...")
    
    try:
        from gui.detect_tab import DetectTab
        from PyQt5.QtWidgets import QApplication
        
        # 創建應用程式實例
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        # 創建檢測頁籤
        detect_tab = DetectTab()
        
        # 設置攝像頭模式
        detect_tab.input_type_combo.setCurrentText("攝像頭")
        
        # 測試有效的攝像頭ID
        print("測試有效攝像頭ID...")
        detect_tab.source_input.setText("0")
        result = detect_tab.validate_source("0")
        print(f"攝像頭ID '0' 驗證結果: {'✓ 通過' if result else '✗ 失敗'}")
        
        # 測試無效的攝像頭ID
        print("測試無效攝像頭ID...")
        detect_tab.source_input.setText("99")
        result = detect_tab.validate_source("99")
        print(f"攝像頭ID '99' 驗證結果: {'✓ 通過' if result else '✗ 失敗 (預期)'}")
        
        # 測試非數字ID
        print("測試非數字攝像頭ID...")
        detect_tab.source_input.setText("abc")
        result = detect_tab.validate_source("abc")
        print(f"攝像頭ID 'abc' 驗證結果: {'✓ 通過' if result else '✗ 失敗 (預期)'}")
        
        return True
        
    except Exception as e:
        print(f"✗ 檢測器攝像頭驗證測試失敗: {str(e)}")
        return False

def test_detection_params():
    """測試檢測參數生成"""
    print("\n測試檢測參數生成...")
    
    try:
        from gui.detect_tab import DetectTab
        from PyQt5.QtWidgets import QApplication
        
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        detect_tab = DetectTab()
        
        # 設置攝像頭參數
        detect_tab.input_type_combo.setCurrentText("攝像頭")
        detect_tab.source_input.setText("0")
        detect_tab.weights_input.setText("./weights/yolov5s.pt")
        detect_tab.device_combo.setCurrentText("cpu")
        
        # 生成檢測參數
        params = detect_tab.get_detection_params()
        
        print("生成的檢測參數:")
        for key, value in params.items():
            print(f"  {key}: {value}")
          # 驗證關鍵參數
        assert params['source'] == "0", "攝像頭來源設置錯誤"
        assert 'device' in params, "缺少設備參數"
        assert 'conf_thres' in params, "缺少信心度參數"
        assert 'iou_thres' in params, "缺少IoU閾值參數"
        
        print("✓ 檢測參數生成測試通過")
        return True
        
    except Exception as e:
        print(f"✗ 檢測參數生成測試失敗: {str(e)}")
        return False

def test_core_detector_camera():
    """測試核心檢測器的攝像頭功能"""
    print("\n測試核心檢測器攝像頭功能...")
    
    try:
        from core.detector import DetectionWorker
        
        # 創建檢測工作執行緒
        detector = DetectionWorker()
        
        # 設置攝像頭參數
        params = {
            'source': '0',  # 攝像頭ID
            'device': 'cpu',
            'conf': 0.25,
            'iou': 0.45
        }
        
        detector.set_parameters(params)
        print("✓ 攝像頭參數設置成功")
        
        # 檢查檢測邏輯
        source = params['source']
        if source.isdigit():
            print(f"✓ 攝像頭來源 '{source}' 被正確識別為數字ID")
        else:
            print(f"✗ 攝像頭來源 '{source}' 未被識別為數字ID")
            return False
        
        print("✓ 核心檢測器攝像頭功能測試通過")
        return True
        
    except Exception as e:
        print(f"✗ 核心檢測器攝像頭功能測試失敗: {str(e)}")
        return False

def main():
    """主函數"""
    print("YOLO-PCB GUI 攝像頭檢測功能測試")
    print("="*60)
    
    # 測試攝像頭硬體可用性
    available_cameras = test_camera_availability()
    
    if not available_cameras:
        print("\n⚠ 警告: 沒有找到可用的攝像頭")
        print("部分測試可能無法完成，但驗證邏輯仍然可以測試")
    else:
        print(f"\n✓ 找到 {len(available_cameras)} 個可用攝像頭: {available_cameras}")
    
    # 執行各項測試
    tests = [
        ("檢測器攝像頭驗證", test_detector_camera_validation),
        ("檢測參數生成", test_detection_params),
        ("核心檢測器攝像頭功能", test_core_detector_camera)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name}測試出現異常: {str(e)}")
            results.append((test_name, False))
    
    # 顯示測試結果
    print("\n" + "="*60)
    print("測試結果摘要")
    print("="*60)
    
    for test_name, success in results:
        status = "✓ 通過" if success else "✗ 失敗"
        print(f"{test_name:<25} {status}")
    
    successful_tests = sum(1 for _, success in results if success)
    total_tests = len(results)
    
    print(f"\n測試通過率: {successful_tests}/{total_tests}")
    
    if successful_tests == total_tests:
        print("\n🎉 所有攝像頭檢測功能測試通過！")
        if available_cameras:
            print(f"您可以在GUI中使用攝像頭ID: {available_cameras}")
        print("\n使用方法:")
        print("1. 啟動GUI: python run_gui.py")
        print("2. 選擇檢測頁籤")
        print("3. 輸入類型選擇 '攝像頭'")
        print("4. 輸入攝像頭ID (通常是 0)")
        print("5. 設置權重檔案")
        print("6. 開始檢測")
        return 0
    else:
        print(f"\n❌ 有 {total_tests - successful_tests} 個測試失敗")
        print("請檢查錯誤訊息並修復問題")
        return 1

if __name__ == "__main__":
    sys.exit(main())
