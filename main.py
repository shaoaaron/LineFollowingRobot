import cv2
import numpy as np

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        line = max(contours, key=cv2.contourArea)
        M = cv2.moments(line)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])  # Centroid X
            return cx
    return None

def pid_control(cx, frame_center, prev_error, Kp=0.1, Ki=0.01, Kd=0.5):
    error = cx - frame_center
    P = Kp * error
    I = Ki * (error + prev_error)
    D = Kd * (error - prev_error)
    control = P + I + D
    return control, error

