"""
測試簡化的YOLOv5載入器功能
"""

import sys
import os
from pathlib import Path

# 添加專案路徑到 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from yolo_gui_utils.simple_yolo_loader_v2 import YOLOv5Loader


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
        
        # 嘗試取得模型參數資訊
        try:
            if hasattr(model, 'parameters'):
                params = list(model.parameters())
                print(f"   參數數量: {len(params)}")
                if params:
                    print(f"   第一個參數設備: {params[0].device}")
                    print(f"   第一個參數形狀: {params[0].shape}")
        except Exception as e:
            print(f"   參數資訊取得失敗: {e}")
        
        return True
    else:
        print(f"❌ 載入失敗: {error}")
        return False


def test_other_weights():
    """測試其他權重檔案"""
    print("\n測試其他權重檔案...")
    
    weights_dir = Path(r"C:/SourceCode/JPC/YOLO-PCB/weights")
    if not weights_dir.exists():
        print("❌ 權重目錄不存在")
        return False
    
    # 尋找所有.pt檔案
    weight_files = list(weights_dir.glob("*.pt"))
    
    if not weight_files:
        print("❌ 沒有找到權重檔案")
        return False
    
    print(f"找到 {len(weight_files)} 個權重檔案:")
    for weight_file in weight_files:
        print(f"  - {weight_file.name}")
    
    # 測試第一個權重檔案
    test_file = weight_files[0]
    loader = YOLOv5Loader()
    success, model, error = loader.load_model(str(test_file), 'cpu')
    
    if success:
        print(f"✅ 成功載入第一個權重檔案: {test_file.name}")
        return True
    else:
        print(f"❌ 載入第一個權重檔案失敗: {error}")
        return False


def main():
    """主測試函數"""
    print("簡化YOLOv5載入器測試")
    print("=" * 50)
    
    # 測試結果
    results = []
    
    # 測試baseline權重
    results.append(test_baseline_weights())
    
    # 測試其他權重
    results.append(test_other_weights())
    
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
