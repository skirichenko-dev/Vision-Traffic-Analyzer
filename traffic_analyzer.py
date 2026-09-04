import cv2
import numpy as np

def analyze_traffic(video_source):
    print(f"[*] Инициализация видеопотока: {video_source}")
    cap = cv2.VideoCapture(video_source)
    
    object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        roi = frame[int(height/2):height, 0:width]

        mask = object_detector.apply(roi)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cap.release()
    cv2.destroyAllWindows()
    print("[+] Анализ видеопотока завершен.")

if __name__ == "__main__":
    print("Vision-Traffic-Analyzer v1.0")
    print("Ожидание видеопотока...")
