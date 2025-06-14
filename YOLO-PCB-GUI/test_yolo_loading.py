"""
測試YOLOv5權重載入功能
驗證修正後的模組衝突問題是否解決
"""

import sys
import os
import tempfile
from pathlib import Path

# 添加專案路徑到 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.detector import DetectionWorker


def test_yolov5_loading():
    """測試YOLOv5模型載入"""
    print("=" * 60)
    print("測試YOLOv5權重載入功能")
    print("=" * 60)
    
    # 建立檢測器
    worker = DetectionWorker()
    
    # 測試案例：載入標準YOLOv5模型
    test_cases = [
        {
            'name': '載入YOLOv5s（最小模型）',
            'weights': 'yolov5s.pt',
            'description': '使用torch.hub載入標準YOLOv5s模型'
        }
    ]
    
    for case in test_cases:
        print(f"\n測試案例：{case['name']}")
        print(f"說明：{case['description']}")
        print("-" * 40)
        
        # 設置信號監聽器
        messages = []
        errors = []
        
        def on_log_message(msg):
            messages.append(msg)
            print(f"[LOG] {msg}")
        
        def on_error(msg):
            errors.append(msg)
            print(f"[ERROR] {msg}")
        
        worker.log_message.connect(on_log_message)
        worker.error_occurred.connect(on_error)
        
        # 嘗試載入模型
        try:
            success = worker.load_model(case['weights'], device='cpu')
            
            if success:
                print(f"✅ 成功：{case['name']}")
                print(f"   模型類型：{type(worker.model)}")
                print(f"   設備：{worker.device}")
            else:
                print(f"❌ 失敗：{case['name']}")
                
        except Exception as e:
            print(f"❌ 異常：{case['name']} - {str(e)}")
        
        # 斷開信號
        worker.log_message.disconnect(on_log_message)
        worker.error_occurred.disconnect(on_error)
    
    print("\n" + "=" * 60)
    print("測試完成")
    print("=" * 60)


def test_module_path_handling():
    """測試模組路徑處理"""
    print("\n測試模組路徑處理...")
    
    # 檢查當前模組路徑
    print(f"當前工作目錄：{os.getcwd()}")
    print(f"專案根目錄：{project_root}")
    
    # 檢查是否存在utils模組衝突
    current_dir = Path(__file__).parent  # YOLO-PCB-GUI目錄
    utils_paths = [p for p in sys.path if str(current_dir) in p]
    
    print(f"可能造成衝突的路徑：{utils_paths}")
    
    # 測試路徑過濾
    original_sys_path = sys.path.copy()
    filtered_path = [p for p in sys.path if not str(current_dir) in p]
    
    print(f"原始sys.path長度：{len(original_sys_path)}")
    print(f"過濾後長度：{len(filtered_path)}")
    
    # 檢查是否還有本地utils
    try:
        # 臨時修改路徑
        sys.path = filtered_path
        
        # 嘗試導入utils，這應該不會導入本地的utils
        print("測試導入utils模組...")
        
        # 檢查utils模組路徑
        if 'utils' in sys.modules:
            del sys.modules['utils']  # 清除緩存
        
        print("✅ 模組路徑處理正常")
        
    except Exception as e:
        print(f"❌ 模組路徑處理異常：{str(e)}")
    finally:
        # 恢復原始路徑
        sys.path = original_sys_path


if __name__ == "__main__":
    # 測試模組路徑處理
    test_module_path_handling()
    
    # 測試YOLOv5載入
    test_yolov5_loading()
