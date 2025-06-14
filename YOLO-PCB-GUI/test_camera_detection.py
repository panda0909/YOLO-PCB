"""
測試攝像頭檢測功能
驗證即時畫面顯示和停止功能
"""

import sys
import os
import cv2
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

# 添加專案路徑到 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from gui.detect_tab import DetectTab


def test_camera_detection():
    """測試攝像頭檢測功能"""
    print("測試攝像頭檢測功能")
    print("=" * 50)
    
    # 檢查攝像頭可用性
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 攝像頭不可用，無法測試")
        return False
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ 攝像頭無法讀取幀")
        return False
    
    print(f"✅ 攝像頭可用，幀尺寸: {frame.shape}")
    
    # 檢查權重檔案
    weights_path = r"C:/SourceCode/JPC/YOLO-PCB/weights/baseline_fpn.pt"
    if not os.path.exists(weights_path):
        print("❌ 權重檔案不存在")
        return False
    
    print(f"✅ 權重檔案存在: {weights_path}")
    
    # 創建GUI應用程式
    app = QApplication(sys.argv if len(sys.argv) > 1 else [])
    
    # 創建檢測頁籤
    detect_tab = DetectTab()
      # 設置檢測參數
    detect_tab.source_input.setText("0")  # 攝像頭
    detect_tab.weights_input.setText(weights_path)
    detect_tab.conf_spinbox.setValue(0.25)
    detect_tab.iou_spinbox.setValue(0.45)
    
    print("✅ 檢測頁籤已創建並設置參數")
    
    # 顯示視窗
    detect_tab.show()
    detect_tab.resize(1200, 800)
    
    print("✅ GUI視窗已顯示")
    
    # 創建自動測試邏輯
    test_sequence = TestSequence(detect_tab)
    
    # 啟動應用程式
    app.exec_()
    
    return True


class TestSequence:
    """自動測試序列"""
    
    def __init__(self, detect_tab):
        self.detect_tab = detect_tab
        self.step = 0
        
        # 連接信號
        self.detect_tab.detection_started.connect(self.on_detection_started)
        
        # 創建定時器來執行測試步驟
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_step)
        self.timer.start(3000)  # 每3秒執行一步
        
        print("\n🎯 自動測試序列已啟動")
        print("   1. 3秒後自動開始檢測")
        print("   2. 10秒後自動停止檢測")
        print("   3. 您也可以手動點擊按鈕測試")
    
    def next_step(self):
        """執行下一個測試步驟"""
        if self.step == 0:
            # 第一步：開始檢測
            print("\n🚀 步驟1: 自動開始攝像頭檢測...")
            self.detect_tab.start_detection()
            self.step += 1
            self.timer.start(10000)  # 10秒後停止
            
        elif self.step == 1:
            # 第二步：停止檢測
            print("\n🛑 步驟2: 自動停止檢測...")
            self.detect_tab.stop_detection()
            self.step += 1
            self.timer.start(5000)  # 5秒後結束
            
        elif self.step == 2:
            # 第三步：結束測試
            print("\n✅ 自動測試完成！")
            print("   您可以繼續手動測試或關閉視窗")
            self.timer.stop()
    
    def on_detection_started(self):
        """檢測開始時的回調"""
        print("✅ 檢測已成功啟動！")
        print("   - 應該能看到攝像頭畫面")
        print("   - 停止按鈕應該已啟用")
        print("   - 進度條應該顯示為無限模式")


def main():
    """主函數"""
    print("YOLO-PCB 攝像頭檢測功能測試")
    print("=" * 60)
    
    try:
        success = test_camera_detection()
        
        if success:
            print("\n🎉 攝像頭檢測功能測試完成！")
        else:
            print("\n❌ 攝像頭檢測功能測試失敗")
            
    except Exception as e:
        print(f"\n❌ 測試過程中發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
