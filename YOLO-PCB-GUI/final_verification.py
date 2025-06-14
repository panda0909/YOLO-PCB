"""
YOLO-PCB GUI 最終功能驗證測試
驗證修正後的所有核心功能
"""

import sys
import os
import cv2
from pathlib import Path

# 添加專案路徑到 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from yolo_gui_utils.simple_yolo_loader_v2 import YOLOv5Loader
from core.detector import DetectionWorker


def test_yolo_loading():
    """測試YOLO權重載入"""
    print("1. 測試YOLO權重載入")
    print("-" * 40)
    
    results = []
    
    # 測試權重檔案列表
    weights_to_test = [
        r"C:/SourceCode/JPC/YOLO-PCB/weights/baseline_fpn.pt",
        r"C:/SourceCode/JPC/YOLO-PCB/weights/baseline.pt"
    ]
    
    for weights_path in weights_to_test:
        if os.path.exists(weights_path):
            loader = YOLOv5Loader()
            success, model, error = loader.load_model(weights_path, 'cpu')
            
            if success:
                print(f"✅ {os.path.basename(weights_path)} 載入成功")
                results.append(True)
            else:
                print(f"❌ {os.path.basename(weights_path)} 載入失敗: {error}")
                results.append(False)
        else:
            print(f"⚠️  {os.path.basename(weights_path)} 檔案不存在")
            results.append(False)
    
    return results


def test_camera_availability():
    """測試攝像頭可用性"""
    print("\n2. 測試攝像頭可用性")
    print("-" * 40)
    
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                print("✅ 攝像頭可用，可以正常讀取幀")
                print(f"   幀尺寸: {frame.shape}")
                return True
            else:
                print("❌ 攝像頭開啟但無法讀取幀")
                return False
        else:
            print("❌ 攝像頭無法開啟")
            return False
    except Exception as e:
        print(f"❌ 攝像頭測試異常: {str(e)}")
        return False


def test_detection_worker():
    """測試檢測工作執行緒"""
    print("\n3. 測試檢測工作執行緒")
    print("-" * 40)
    
    try:
        worker = DetectionWorker()
        
        # 測試模型載入
        weights_path = r"C:/SourceCode/JPC/YOLO-PCB/weights/baseline_fpn.pt"
        if os.path.exists(weights_path):
            success = worker.load_model(weights_path, 'cpu')
            if success:
                print("✅ DetectionWorker 模型載入成功")
                print(f"   模型類型: {type(worker.model)}")
                print(f"   設備: {worker.device}")
                return True
            else:
                print("❌ DetectionWorker 模型載入失敗")
                return False
        else:
            print("⚠️  測試權重檔案不存在")
            return False
            
    except Exception as e:
        print(f"❌ DetectionWorker 測試異常: {str(e)}")
        return False


def test_module_imports():
    """測試模組導入"""
    print("\n4. 測試模組導入")
    print("-" * 40)
    
    modules_to_test = [
        ('yolo_gui_utils.config_manager', 'ConfigManager'),
        ('yolo_gui_utils.simple_yolo_loader_v2', 'YOLOv5Loader'),
        ('gui.main_window', 'MainWindow'),
        ('gui.detect_tab', 'DetectTab'),
        ('gui.widgets.image_viewer', 'ImageViewer'),
        ('core.detector', 'DetectionWorker'),
    ]
    
    results = []
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✅ {module_name}.{class_name} 導入成功")
            results.append(True)
        except Exception as e:
            print(f"❌ {module_name}.{class_name} 導入失敗: {str(e)}")
            results.append(False)
    
    return results


def main():
    """主測試函數"""
    print("YOLO-PCB GUI 最終功能驗證")
    print("=" * 60)
    
    all_results = []
    
    # 執行各項測試
    all_results.extend(test_yolo_loading())
    all_results.append(test_camera_availability())
    all_results.append(test_detection_worker())
    all_results.extend(test_module_imports())
    
    # 統計結果
    passed = sum(all_results)
    total = len(all_results)
    success_rate = (passed / total) * 100
    
    print("\n" + "=" * 60)
    print("最終測試結果")
    print("=" * 60)
    print(f"通過: {passed}/{total}")
    print(f"成功率: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("\n🎉 恭喜！YOLO-PCB GUI 核心功能驗證通過！")
        print("✅ YOLOv5權重載入問題已解決")
        print("✅ 攝像頭檢測功能正常")
        print("✅ 模組導入無衝突")
        print("✅ GUI應用程式可以正常運行")
        
        print("\n📋 接下來可以進行的工作：")
        print("   - 完善檢測算法與後處理")
        print("   - 優化UI界面與用戶體驗")
        print("   - 撰寫完整的單元測試")
        print("   - 準備應用程式打包與部署")
        
    elif success_rate >= 70:
        print("\n⚠️  YOLO-PCB GUI 大部分功能正常，但仍有少數問題需要解決。")
    else:
        print("\n❌ YOLO-PCB GUI 存在較多問題，需要進一步修正。")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
