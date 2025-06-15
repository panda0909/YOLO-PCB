#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
測試新的檢測頁籤佈局
"""

import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt

# 添加GUI路徑
gui_path = os.path.join(os.path.dirname(__file__), 'gui')
sys.path.insert(0, gui_path)

try:
    from detect_tab import DetectTab
    
    class TestWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("YOLO-PCB 檢測頁籤佈局測試")
            self.setGeometry(100, 100, 1400, 800)  # 設置窗口大小
            
            # 創建中央部件
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            
            # 創建佈局
            layout = QVBoxLayout(central_widget)
            
            # 添加檢測頁籤
            self.detect_tab = DetectTab(self)
            layout.addWidget(self.detect_tab)
            
            # 設置樣式
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #f0f0f0;
                }
                QGroupBox {
                    font-weight: bold;
                    border: 2px solid #cccccc;
                    border-radius: 5px;
                    margin-top: 1ex;
                    padding-top: 10px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                }
            """)
    
    if __name__ == "__main__":
        app = QApplication(sys.argv)
        
        # 設置應用程式樣式
        app.setStyle('Fusion')
        
        window = TestWindow()
        window.show()
        
        print("新佈局預覽:")
        print("- 左側面板：包含所有控制項目（40%寬度）")
        print("- 右側面板：圖像顯示區域（60%寬度）")
        print("- 圖像顯示區域最小尺寸：800x600")
        print("- 左側面板最大寬度：650px")
        print("")
        print("可以調整窗口大小來測試響應式佈局")
        
        sys.exit(app.exec_())
        
except ImportError as e:
    print(f"導入錯誤: {e}")
    print("請確保所有必要的模組都已正確安裝")
    print("特別是PyQt5和相關的GUI組件")
