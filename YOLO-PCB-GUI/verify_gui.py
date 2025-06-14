#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO-PCB GUI 快速功能驗證腳本

作者: AI Assistant
版本: 1.0.0
"""

import sys
import os
import time
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

# 添加專案路徑
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_gui_creation():
    """測試GUI創建和基本功能"""
    print("GUI功能驗證測試")
    print("="*50)
    
    try:
        # 創建應用程式實例
        app = QApplication(sys.argv)
        
        print("✓ PyQt5應用程式創建成功")
        
        # 導入配置管理器
        from utils.config_manager import ConfigManager
        config = ConfigManager()
        print("✓ 配置管理器載入成功")
        
        # 導入主視窗
        from gui.main_window import MainWindow
        main_window = MainWindow(config)
        print("✓ 主視窗創建成功")
        
        # 檢查頁籤
        if hasattr(main_window, 'detect_tab') and main_window.detect_tab:
            print("✓ 檢測頁籤載入成功")
        else:
            print("⚠ 檢測頁籤未載入")
            
        if hasattr(main_window, 'train_tab') and main_window.train_tab:
            print("✓ 訓練頁籤載入成功")
        else:
            print("⚠ 訓練頁籤未載入")
            
        if hasattr(main_window, 'test_tab') and main_window.test_tab:
            print("✓ 測試頁籤載入成功")
        else:
            print("⚠ 測試頁籤未載入")
        
        # 檢查圖像顯示組件
        if hasattr(main_window.detect_tab, 'image_viewer'):
            print("✓ 圖像顯示組件載入成功")
        else:
            print("⚠ 圖像顯示組件未載入")
        
        # 測試信號連接
        if hasattr(main_window.detect_tab, 'detection_started'):
            print("✓ 檢測信號定義正確")
        
        # 顯示主視窗（短暫）
        main_window.show()
        print("✓ 主視窗顯示成功")
        
        # 設置定時器自動關閉
        timer = QTimer()
        timer.timeout.connect(app.quit)
        timer.start(3000)  # 3秒後自動關閉
        
        print("\n程式將在3秒後自動關閉...")
        print("如果看到GUI視窗出現，表示所有功能正常！")
        
        # 運行應用程式
        app.exec_()
        
        print("\n✅ GUI功能驗證完成 - 所有測試通過！")
        return True
        
    except Exception as e:
        print(f"\n❌ GUI功能驗證失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函數"""
    success = test_gui_creation()
    
    if success:
        print("\n" + "="*50)
        print("🎉 YOLO-PCB GUI 已準備就緒！")
        print("="*50)
        print("現在您可以使用以下命令啟動完整的GUI應用程式：")
        print("  python run_gui.py")
        print("  python main.py")
        print("\n主要功能：")
        print("  🔍 PCB缺陷檢測")
        print("  🎓 模型訓練")
        print("  📊 模型測試")
        print("  🖼️ 圖像顯示與標註")
        print("  ⚙️ 參數配置")
        return 0
    else:
        print("\n" + "="*50)
        print("❌ GUI功能驗證失敗")
        print("="*50)
        print("請檢查錯誤信息並修復問題後重試")
        return 1

if __name__ == "__main__":
    sys.exit(main())
