"""
測試新的YOLOv5載入器功能
"""

import sys
import os
from pathlib import Path

# 添加專案路徑到 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from yolo_gui_utils.yolo_loader import YOLOv5Loader


def test_baseline_weights():
    """測試baseline_fpn.pt權重載入"""
    print("測試YOLO-PCB自定義權重載入...")
    
    # 權重檔案路徑
    weights_path = r"C:/SourceCode/JPC/YOLO-PCB/weights/baseline_fpn.pt"
    
    if not os.path.exists(weights_path):
        print(f"❌ 權重檔案不存在: {weights_path}")
        return False
    
    # 建立載入器
    loader = YOLOv5Loader()
    
    # 嘗試載入
    success, model, error = loader.load_model(weights_path, 'cpu')
    
    if success:
        print(f"✅ 成功載入權重: {weights_path}")
        print(f"   模型類型: {type(model)}")
        print(f"   模型設備: {next(model.parameters()).device if hasattr(model, 'parameters') else 'N/A'}")
        return True
    else:
        print(f"❌ 載入失敗: {error}")
        return False


def test_standard_yolo():
    """測試標準YOLOv5權重載入"""
    print("\n測試標準YOLOv5權重載入...")
    
    loader = YOLOv5Loader()
    success, model, error = loader.load_model('yolov5s.pt', 'cpu')
    
    if success:
        print("✅ 標準YOLOv5s載入成功")
        print(f"   模型類型: {type(model)}")
        return True
    else:
        print(f"❌ 標準YOLOv5s載入失敗: {error}")
        return False


def main():
    """主測試函數"""
    print("YOLOv5載入器測試")
    print("=" * 50)
    
    # 測試結果
    results = []
    
    # 測試標準權重
    results.append(test_standard_yolo())
    
    # 測試自定義權重
    results.append(test_baseline_weights())
    
    # 總結
    print("\n" + "=" * 50)
    print("測試總結:")
    print(f"通過: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 所有測試通過！")
    else:
        print("⚠️  有測試失敗，請檢查相關問題。")


if __name__ == "__main__":
    main()
