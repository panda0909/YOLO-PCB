"""
完整功能測試腳本
測試YOLO-PCB GUI的各項核心功能，包括：
1. YOLOv5權重載入
2. 攝像頭檢測
3. 圖像檢測
4. 參數驗證
"""

import sys
import os
import time
import cv2
import tempfile
from pathlib import Path
from PyQt5.QtCore import QCoreApplication

# 添加專案路徑到 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.detector import DetectionWorker
from gui.detect_tab import DetectTab
from yolo_gui_utils.config_manager import ConfigManager


class TestResults:
    """測試結果記錄器"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def add_test(self, name, success, message=""):
        self.tests.append({
            'name': name,
            'success': success,
            'message': message
        })
        if success:
            self.passed += 1
            print(f"✅ {name}")
        else:
            self.failed += 1
            print(f"❌ {name}: {message}")
    
    def print_summary(self):
        print("\n" + "=" * 60)
        print("測試結果總結")
        print("=" * 60)
        print(f"通過: {self.passed}")
        print(f"失敗: {self.failed}")
        print(f"總計: {len(self.tests)}")
        print("=" * 60)
        
        if self.failed > 0:
            print("\n失敗的測試：")
            for test in self.tests:
                if not test['success']:
                    print(f"  - {test['name']}: {test['message']}")


def test_yolo_loading():
    """測試YOLOv5權重載入"""
    print("測試YOLOv5權重載入...")
    results = TestResults()
    
    worker = DetectionWorker()
    
    # 測試1: YOLOv5s載入
    try:
        success = worker.load_model('yolov5s.pt', 'cpu')
        results.add_test("YOLOv5s模型載入", success)
        
        if success:
            # 測試模型基本屬性
            results.add_test("模型設備設定", worker.device.type == 'cpu')
            results.add_test("模型物件存在", worker.model is not None)
            
    except Exception as e:
        results.add_test("YOLOv5s模型載入", False, str(e))
    
    return results


def test_camera_detection():
    """測試攝像頭檢測功能"""
    print("\n測試攝像頭檢測功能...")
    results = TestResults()
    
    # 檢查攝像頭可用性
    cap = cv2.VideoCapture(0)
    camera_available = cap.isOpened()
    cap.release()
    
    results.add_test("攝像頭可用性檢查", camera_available)
    
    if camera_available:
        # 測試檢測參數生成
        try:
            app = QCoreApplication.instance()
            if app is None:
                app = QCoreApplication([])
            
            detect_tab = DetectTab()
            
            # 設定攝像頭輸入
            detect_tab.source_input.setText("0")
            detect_tab.weights_input.setText("yolov5s.pt")
            
            # 生成檢測參數
            params = detect_tab.get_detection_parameters()
            
            # 驗證參數
            results.add_test("攝像頭參數生成", 'source' in params and params['source'] == '0')
            results.add_test("權重參數設定", 'weights' in params and params['weights'] == 'yolov5s.pt')
            results.add_test("來源驗證功能", hasattr(detect_tab, 'validate_source'))
            
            # 測試來源驗證
            is_valid, error_msg = detect_tab.validate_source('0')
            results.add_test("攝像頭來源驗證", is_valid, error_msg)
            
        except Exception as e:
            results.add_test("攝像頭檢測參數", False, str(e))
    
    return results


def test_image_detection():
    """測試圖像檢測功能"""
    print("\n測試圖像檢測功能...")
    results = TestResults()
    
    # 創建測試圖像
    test_image_path = None
    try:
        # 創建一個簡單的測試圖像
        import numpy as np
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        test_img[100:200, 100:200] = [255, 0, 0]  # 紅色方塊
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            test_image_path = tmp_file.name
            cv2.imwrite(test_image_path, test_img)
        
        results.add_test("測試圖像創建", os.path.exists(test_image_path))
        
        # 測試圖像檢測
        worker = DetectionWorker()
        
        # 載入模型
        model_loaded = worker.load_model('yolov5s.pt', 'cpu')
        results.add_test("檢測器模型載入", model_loaded)
        
        if model_loaded:
            # 設定檢測參數
            params = {
                'source': test_image_path,
                'weights': 'yolov5s.pt',
                'device': 'cpu',
                'conf_threshold': 0.25,
                'iou_threshold': 0.45,
                'max_det': 1000,
                'output': None
            }
            
            worker.set_parameters(params)
            results.add_test("檢測參數設定", True)
            
            # 測試推理方法
            try:
                test_frame = cv2.imread(test_image_path)
                inference_result = worker._inference(test_frame)
                results.add_test("圖像推理執行", True)
                results.add_test("推理結果格式", isinstance(inference_result, list))
                
            except Exception as e:
                results.add_test("圖像推理執行", False, str(e))
        
    except Exception as e:
        results.add_test("圖像檢測測試", False, str(e))
    
    finally:
        # 清理測試文件
        if test_image_path and os.path.exists(test_image_path):
            try:
                os.unlink(test_image_path)
            except:
                pass
    
    return results


def test_config_management():
    """測試配置管理功能"""
    print("\n測試配置管理功能...")
    results = TestResults()
    
    try:
        config_manager = ConfigManager()
        
        # 測試配置載入
        results.add_test("配置管理器初始化", True)
        
        # 測試配置讀取
        test_key = "test_setting"
        test_value = "test_value"
        
        config_manager.set(test_key, test_value)
        retrieved_value = config_manager.get(test_key)
        
        results.add_test("配置設定與讀取", retrieved_value == test_value)
        
        # 測試配置保存
        config_manager.save()
        results.add_test("配置保存", True)
        
    except Exception as e:
        results.add_test("配置管理測試", False, str(e))
    
    return results


def main():
    """主測試函數"""
    print("YOLO-PCB GUI 完整功能測試")
    print("=" * 60)
    
    all_results = []
    
    # 執行各項測試
    all_results.append(test_yolo_loading())
    all_results.append(test_camera_detection())
    all_results.append(test_image_detection())
    all_results.append(test_config_management())
    
    # 統計總結果
    total_passed = sum(r.passed for r in all_results)
    total_failed = sum(r.failed for r in all_results)
    total_tests = sum(len(r.tests) for r in all_results)
    
    print("\n" + "=" * 60)
    print("整體測試結果")
    print("=" * 60)
    print(f"通過: {total_passed}")
    print(f"失敗: {total_failed}")
    print(f"總計: {total_tests}")
    print(f"成功率: {(total_passed/total_tests*100):.1f}%")
    print("=" * 60)
    
    # 顯示詳細結果
    for i, result in enumerate(all_results):
        if result.failed > 0:
            print(f"\n第{i+1}組測試失敗詳情：")
            for test in result.tests:
                if not test['success']:
                    print(f"  - {test['name']}: {test['message']}")
    
    if total_failed == 0:
        print("\n🎉 所有測試通過！YOLO-PCB GUI 功能正常！")
    else:
        print(f"\n⚠️  有 {total_failed} 項測試失敗，請檢查相關功能。")


if __name__ == "__main__":
    main()
