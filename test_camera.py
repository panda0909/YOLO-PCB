# 攝影機診斷腳本 - camera_test.py
import cv2
import sys

def test_camera(camera_id=0):
    print(f"測試攝影機 {camera_id}...")
    
    # 嘗試開啟攝影機
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ 攝影機 {camera_id} 無法開啟")
        return False
    
    print(f"✅ 攝影機 {camera_id} 開啟成功")
    
    # 測試讀取畫面
    ret, frame = cap.read()
    if not ret:
        print("❌ 無法讀取攝影機畫面")
        cap.release()
        return False
    
    print(f"✅ 畫面讀取成功，解析度: {frame.shape[:2]}")
    
    # 顯示測試畫面
    cv2.imshow(f'Camera {camera_id} Test', frame)
    print("✅ 測試視窗已開啟，按任意鍵關閉...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()
    
    return True

# 測試攝影機 0 到 3
for i in range(4):
    if test_camera(i):
        print(f"✅ 攝影機 {i} 可用於檢測")
        break
else:
    print("❌ 未找到可用的攝影機")