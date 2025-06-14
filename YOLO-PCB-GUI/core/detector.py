"""
檢測器核心模組
提供YOLO-PCB檢測功能的核心實現
"""

import os
import sys
import time
import cv2
import torch
import numpy as np
from pathlib import Path
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

# 導入自定義的YOLO載入器
from yolo_gui_utils.simple_yolo_loader_v2 import YOLOv5Loader


class DetectionWorker(QThread):
    """檢測工作執行緒"""
    
    # 信號定義
    frame_processed = pyqtSignal(np.ndarray)  # 處理完的幀
    detection_result = pyqtSignal(dict)       # 檢測結果
    progress_updated = pyqtSignal(int)        # 進度更新
    log_message = pyqtSignal(str)             # 日誌訊息
    detection_finished = pyqtSignal(str)      # 檢測完成
    error_occurred = pyqtSignal(str)          # 錯誤發生
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.device = None
        self.params = {}
        self.running = False
        self.camera = None
        
    def set_parameters(self, params):
        """設置檢測參數"""
        self.params = params.copy()
        
    def load_model(self, weights_path, device='auto'):
        """載入YOLO模型"""
        try:
            self.log_message.emit(f"正在載入模型: {weights_path}")
            
            # 設置設備
            if device == 'auto':
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(device)
                
            self.log_message.emit(f"使用設備: {self.device}")
            
            # 檢查是否為YOLOv5模型
            if weights_path.endswith('.pt'):
                # 使用專門的YOLOv5載入器
                loader = YOLOv5Loader()
                success, model, error = loader.load_model(weights_path, str(self.device))
                
                if success:
                    self.model = model
                    self.log_message.emit("模型載入成功")
                    return True
                else:
                    self.log_message.emit(f"模型載入失敗: {error}")
                    self.error_occurred.emit(f"模型載入失敗: {error}")
                    return False
                    
            else:
                self.error_occurred.emit(f"不支援的模型格式: {weights_path}")
                return False
                
        except Exception as e:
            error_msg = f"模型載入失敗: {str(e)}"
            self.log_message.emit(error_msg)
            self.error_occurred.emit(error_msg)
            return False
    
    def run(self):
        """執行檢測"""
        try:
            if not self.model:
                self.error_occurred.emit("模型未載入")
                return
            
            source = self.params.get('source', '')
            
            if not source:
                self.error_occurred.emit("未指定輸入來源")
                return
            
            # 根據來源類型執行不同的檢測
            if source.isdigit():  # 攝像頭
                self._detect_camera(int(source))
            elif os.path.isfile(source):
                if source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    self._detect_image(source)
                elif source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    self._detect_video(source)
                else:
                    self.error_occurred.emit(f"不支援的檔案格式: {source}")
            elif os.path.isdir(source):
                self._detect_folder(source)
            else:
                self.error_occurred.emit(f"無效的輸入來源: {source}")
                
        except Exception as e:
            self.error_occurred.emit(f"檢測執行失敗: {str(e)}")
    
    def _detect_image(self, image_path):
        """檢測單張圖片"""
        try:
            self.log_message.emit(f"正在檢測圖片: {image_path}")
            
            # 讀取圖片
            img = cv2.imread(image_path)
            if img is None:
                self.error_occurred.emit(f"無法讀取圖片: {image_path}")
                return
            
            # 執行檢測
            results = self._inference(img)
            
            # 繪製結果
            annotated_img = self._draw_results(img, results)
            
            # 發送結果
            self.frame_processed.emit(annotated_img)
            self.detection_result.emit({
                'image_path': image_path,
                'detections': results,
                'count': len(results) if results else 0
            })
            
            # 保存結果
            if self.params.get('output'):
                self._save_results(image_path, annotated_img, results)
            
            self.detection_finished.emit(self.params.get('output', ''))
            
        except Exception as e:
            self.error_occurred.emit(f"圖片檢測失敗: {str(e)}")
    
    def _detect_video(self, video_path):
        """檢測視頻"""
        try:
            self.log_message.emit(f"正在檢測視頻: {video_path}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.error_occurred.emit(f"無法開啟視頻: {video_path}")
                return
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0
            
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 執行檢測
                results = self._inference(frame)
                
                # 繪製結果
                annotated_frame = self._draw_results(frame, results)
                
                # 發送結果
                self.frame_processed.emit(annotated_frame)
                self.detection_result.emit({
                    'frame_index': frame_count,
                    'detections': results,
                    'count': len(results) if results else 0
                })
                
                # 更新進度
                frame_count += 1
                progress = int((frame_count / total_frames) * 100)
                self.progress_updated.emit(progress)
                
                # 控制幀率
                self.msleep(33)  # ~30 FPS
            
            cap.release()
            self.detection_finished.emit(self.params.get('output', ''))
            
        except Exception as e:
            self.error_occurred.emit(f"視頻檢測失敗: {str(e)}")
    
    def _detect_camera(self, camera_id):
        """檢測攝像頭"""
        try:
            self.log_message.emit(f"正在開啟攝像頭: {camera_id}")
            
            self.camera = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            if not self.camera.isOpened():
                self.error_occurred.emit(f"無法開啟攝像頭: {camera_id}")
                return
            
            frame_count = 0
            
            while self.running and self.camera.isOpened():
                ret, frame = self.camera.read()
                if not ret:
                    self.error_occurred.emit("無法從攝像頭讀取畫面")
                    break
                
                # 執行檢測
                results = self._inference(frame)
                
                # 繪製結果
                annotated_frame = self._draw_results(frame, results)
                
                # 發送結果
                self.frame_processed.emit(annotated_frame)
                self.detection_result.emit({
                    'frame_index': frame_count,
                    'detections': results,
                    'count': len(results) if results else 0
                })
                
                frame_count += 1
                
                # 控制幀率
                self.msleep(33)  # ~30 FPS
            
            if self.camera:
                self.camera.release()
                
        except Exception as e:
            self.error_occurred.emit(f"攝像頭檢測失敗: {str(e)}")
    
    def _detect_folder(self, folder_path):
        """批次檢測資料夾中的圖片"""
        try:
            self.log_message.emit(f"正在批次檢測資料夾: {folder_path}")
            
            # 支援的圖片格式
            supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            
            # 獲取所有圖片檔案
            image_files = []
            for ext in supported_formats:
                image_files.extend(Path(folder_path).glob(f'*{ext}'))
                image_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))
            
            if not image_files:
                self.error_occurred.emit(f"資料夾中未找到支援的圖片檔案: {folder_path}")
                return
            
            total_images = len(image_files)
            self.log_message.emit(f"找到 {total_images} 張圖片")
            
            # 逐個處理圖片
            for i, image_path in enumerate(image_files):
                if not self.running:
                    break
                
                try:
                    # 讀取圖片
                    img = cv2.imread(str(image_path))
                    if img is None:
                        self.log_message.emit(f"跳過無法讀取的圖片: {image_path}")
                        continue
                    
                    # 執行檢測
                    results = self._inference(img)
                    
                    # 繪製結果
                    annotated_img = self._draw_results(img, results)
                    
                    # 發送結果
                    self.frame_processed.emit(annotated_img)
                    self.detection_result.emit({
                        'image_path': str(image_path),
                        'detections': results,
                        'count': len(results) if results else 0
                    })
                    
                    # 保存結果
                    if self.params.get('output'):
                        self._save_results(str(image_path), annotated_img, results)
                    
                    # 更新進度
                    progress = int(((i + 1) / total_images) * 100)
                    self.progress_updated.emit(progress)
                    
                except Exception as e:
                    self.log_message.emit(f"處理圖片失敗 {image_path}: {str(e)}")
                    continue
            
            self.detection_finished.emit(self.params.get('output', ''))
            
        except Exception as e:
            self.error_occurred.emit(f"批次檢測失敗: {str(e)}")
    
    def _inference(self, img):
        """執行模型推理"""
        try:
            # 這裡實現實際的推理邏輯
            # 由於無法確定具體的模型結構，這裡提供一個示例實現
            
            if hasattr(self.model, '__call__'):
                # YOLOv5 hub模型
                results = self.model(img)
                
                # 處理結果
                detections = []
                if hasattr(results, 'pandas'):
                    df = results.pandas().xyxy[0]
                    for _, row in df.iterrows():
                        detection = {
                            'class': row['name'],
                            'confidence': row['confidence'],
                            'bbox': [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
                        }
                        if detection['confidence'] >= self.params.get('conf_thres', 0.25):
                            detections.append(detection)
                
                return detections
            
            else:
                # 其他類型的模型需要實現特定的推理邏輯
                self.log_message.emit("警告: 模型推理方法需要進一步實現")
                return []
                
        except Exception as e:
            self.log_message.emit(f"推理失敗: {str(e)}")
            return []
    
    def _draw_results(self, img, results):
        """在圖片上繪製檢測結果"""
        try:
            annotated_img = img.copy()
            
            if not results:
                return annotated_img
            
            # 繪製參數
            line_thickness = self.params.get('line_thickness', 3)
            hide_labels = self.params.get('hide_labels', False)
            hide_conf = self.params.get('hide_conf', False)
            
            # 類別顏色（PCB缺陷類別）
            colors = {
                'missing_hole': (0, 0, 255),      # 紅色
                'mouse_bite': (0, 255, 0),        # 綠色
                'open_circuit': (255, 0, 0),      # 藍色
                'short': (0, 255, 255),           # 黃色
                'spur': (255, 0, 255),            # 紫色
                'spurious_copper': (255, 255, 0)  # 青色
            }
            
            for detection in results:
                bbox = detection['bbox']
                class_name = detection['class']
                confidence = detection['confidence']
                
                # 獲取顏色
                color = colors.get(class_name, (128, 128, 128))
                
                # 繪製邊界框
                cv2.rectangle(annotated_img, 
                            (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])), 
                            color, line_thickness)
                
                # 繪製標籤
                if not hide_labels or not hide_conf:
                    label = ""
                    if not hide_labels:
                        label += class_name
                    if not hide_conf:
                        if label:
                            label += f" {confidence:.2f}"
                        else:
                            label = f"{confidence:.2f}"
                    
                    # 計算文字大小
                    font_scale = 0.5
                    thickness = 1
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    
                    # 繪製文字背景
                    cv2.rectangle(annotated_img,
                                (int(bbox[0]), int(bbox[1]) - text_height - 10),
                                (int(bbox[0]) + text_width, int(bbox[1])),
                                color, -1)
                    
                    # 繪製文字
                    cv2.putText(annotated_img, label,
                              (int(bbox[0]), int(bbox[1]) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            return annotated_img
            
        except Exception as e:
            self.log_message.emit(f"繪製結果失敗: {str(e)}")
            return img
    
    def _save_results(self, image_path, annotated_img, results):
        """保存檢測結果"""
        try:
            output_dir = self.params.get('output', 'runs/detect')
            os.makedirs(output_dir, exist_ok=True)
            
            # 獲取檔案名稱
            image_name = os.path.basename(image_path)
            name_without_ext = os.path.splitext(image_name)[0]
            
            # 保存標註圖片
            output_image_path = os.path.join(output_dir, image_name)
            cv2.imwrite(output_image_path, annotated_img)
            
            # 保存標註文件
            if self.params.get('save_txt', False):
                txt_path = os.path.join(output_dir, f"{name_without_ext}.txt")
                with open(txt_path, 'w') as f:
                    for detection in results:
                        bbox = detection['bbox']
                        class_name = detection['class']
                        confidence = detection['confidence']
                        
                        # 轉換為YOLO格式（相對坐標）
                        img_h, img_w = annotated_img.shape[:2]
                        x_center = (bbox[0] + bbox[2]) / 2 / img_w
                        y_center = (bbox[1] + bbox[3]) / 2 / img_h
                        width = (bbox[2] - bbox[0]) / img_w
                        height = (bbox[3] - bbox[1]) / img_h
                        
                        # 寫入文件
                        if self.params.get('save_conf', False):
                            f.write(f"{class_name} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {confidence:.6f}\n")
                        else:
                            f.write(f"{class_name} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
        except Exception as e:
            self.log_message.emit(f"保存結果失敗: {str(e)}")
    
    def start_detection(self):
        """開始檢測"""
        self.running = True
        self.start()
    
    def stop_detection(self):
        """停止檢測"""
        self.running = False
        if self.camera:
            self.camera.release()
        if self.isRunning():
            self.quit()
            self.wait(3000)  # 最多等待3秒


class Detector(QObject):
    """檢測器主類別"""
    
    # 信號定義
    detection_started = pyqtSignal()
    detection_finished = pyqtSignal(str)
    log_message = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.worker = None
        
    def start_detection(self, params):
        """開始檢測"""
        try:
            # 創建工作執行緒
            self.worker = DetectionWorker()
            self.worker.set_parameters(params)
            
            # 連接信號
            self.worker.log_message.connect(self.log_message.emit)
            self.worker.detection_finished.connect(self.detection_finished.emit)
            self.worker.error_occurred.connect(self.log_message.emit)
            
            # 載入模型
            if not self.worker.load_model(params['weights'], params.get('device', 'auto')):
                return False
            
            # 開始檢測
            self.worker.start_detection()
            self.detection_started.emit()
            
            return True
            
        except Exception as e:
            self.log_message.emit(f"啟動檢測失敗: {str(e)}")
            return False
    
    def stop_detection(self):
        """停止檢測"""
        if self.worker:
            self.worker.stop_detection()
            self.worker = None
