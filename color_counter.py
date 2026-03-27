import cv2
import numpy as np

# -------------------------------
# 📹 Initialize Webcam
# -------------------------------
cap = cv2.VideoCapture(0)

# -------------------------------
# 🎯 ROI (Region of Interest)
# -------------------------------
x1, y1 = 200, 150
x2, y2 = 450, 350

# -------------------------------
# 🔢 Counter Variables
# -------------------------------
count = 0
object_inside = False   # Current state
prev_state = False      # Previous frame state

# -------------------------------
# 🔵 Color Range (BLUE example)
# -------------------------------
lower_color = np.array([100, 150, 50])
upper_color = np.array([140, 255, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror effect (optional)
    frame = cv2.flip(frame, 1)

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create mask for color detection
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Noise reduction (BONUS improvement)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    object_inside = False  # Reset each frame

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 1500:  # Filter noise
            x, y, w, h = cv2.boundingRect(cnt)

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # -------------------------------
            # 🎯 Trigger Logic
            # Check if object fully inside ROI
            # -------------------------------
            if x > x1 and y > y1 and (x + w) < x2 and (y + h) < y2:
                object_inside = True

    # -------------------------------
    # 🔢 Counting Logic
    # Valid event = ENTER + EXIT
    # -------------------------------
    if object_inside and not prev_state:
        print("Object ENTERED ROI")

    if not object_inside and prev_state:
        print("Object EXITED ROI")
        count += 1   # Increment only on exit

    prev_state = object_inside

    # -------------------------------
    # 🖥️ Display UI Elements
    # -------------------------------
    # Draw ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show count
    cv2.putText(frame, f"Objects Counted: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show window
    cv2.imshow("Color Detection Counter", frame)

    # -------------------------------
    # ❌ Exit Condition
    # -------------------------------
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()