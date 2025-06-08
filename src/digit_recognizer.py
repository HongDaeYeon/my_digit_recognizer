import cv2
import os
import uuid
from predict_digit import predict_digit

video_path = "5.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    roi = frame[h//2 - 100:h//2 + 100, w//2 - 100:w//2 + 100]

    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(temp_filename, roi)

    digit = predict_digit(temp_filename)

    if os.path.exists(temp_filename):
        os.remove(temp_filename)

    if digit is not None:
        cv2.putText(frame, f"Predicted: {digit}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

    cv2.rectangle(frame, (w//2 - 100, h//2 - 100), (w//2 + 100, h//2 + 100), (255, 0, 0), 2)
    cv2.imshow("Digit Recognizer", frame)

    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()