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

frame = np.ones((400, 400, 3), dtype=np.uint8) * 255  # White background
cv2.line(frame, (50, 200), (350, 200), (0, 0, 0), 5)  # Black line

robot_pos = 50
frame_center = frame.shape[1] // 2

prev_error = 0

while True:
    cx = process_frame(frame)  # Line centroid
    if cx != frame_center:
        control, prev_error = pid_control(cx, frame_center, prev_error)
        print(f"Control: {control}")  # Debug control signal
        # Update robot position based on control signal
        robot_pos += int(control * 10)  # Scale control for visible movement
    else:
        print("Robot is centered.")
    
    # Visualize robot
    display_frame = frame.copy()
    cv2.circle(display_frame, (robot_pos, 200), 10, (255, 0, 0), -1)
    
    cv2.imshow("Line Following Simulation", display_frame)
    print(f"Robot Position: {robot_pos}")

    if cv2.waitKey(30) & 0xFF == 27:
        break

cv2.destroyAllWindows()
