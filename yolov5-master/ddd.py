import cv2

for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"摄像头索引为{i}")
        cap.release()
    else:
        print(f"摄像头索引{i}不可用")