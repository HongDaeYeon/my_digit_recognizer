Webcam Digit Recognizer

실시간 웹캠으로 숫자를 인식하는 딥러닝 기반의 숫자 인식기입니다. OpenCV로 실시간 카메라 영상에서 숫자가 포함된 영역을 잘라내고, 전처리 및 딥러닝 모델을 통해 예측 결과를 화면에 표시합니다.

---

개요

이 프로젝트는 TensorFlow 기반 숫자 분류 모델을 이용하여 웹캠의 실시간 영상에서 숫자를 인식합니다. 중심 사각형 영역(ROI)에 손글씨 숫자를 보여주면 실시간으로 예측된 숫자가 화면에 출력됩니다.

---

실행 방법

1. **Python 설치**  
   Python 3.8 이상이 설치되어 있어야 합니다.

2. **필수 패키지 설치**

   ```bash
   pip install opencv-python tensorflow numpy
   ```

3. **실행 명령어**

   ```bash
   python src/webcam_digit_recognizer.py
   ```

4. **사용법**
   - 웹캠이 켜지면 중앙의 파란색 사각형 안에 손글씨 숫자를 보여주세요.
   - 예측된 숫자가 좌측 상단에 실시간으로 표시됩니다.
   - ESC 키를 누르면 종료됩니다.

---

## 주요 기능

- 실시간 웹캠 영상 캡처
- 중심 영역 ROI에서 숫자 영역 추출
- 딥러닝 모델을 통한 숫자 예측
- 예측값 실시간 표시
- 임시 이미지 파일 저장/삭제로 메모리 관리

---

## 디렉터리 구조

```
111/
├── model/
│   └── digit_model.h5           # 사전 학습된 Keras 숫자 분류 모델
├── src/
│   ├── digit_recognizer.py  # 실시간 숫자 인식 실행 파일
│   ├── predict_digit.py            # 숫자 예측 로직 (이미지 입력)
│   └── utils/
│       └── preprocessing.py        # 이미지 전처리 함수
└── README.md                    
```

---

## 주요 파일 설명

### `digit_recognizer.py`

```python
import cv2
import os
import uuid
from predict_digit import predict_digit

cap = cv2.VideoCapture(0)

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

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키
        break

cap.release()
cv2.destroyAllWindows()
```

---

### `predict_digit.py`

```python
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils.preprocessing import preprocess_image
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU 사용 강제

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model/digit_model"))
model = load_model(model_path)

def predict_digit(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "이미지를 불러올 수 없습니다: " + image_path

    processed = preprocess_image(image)
    processed = np.array(processed, dtype=np.float32)
    prediction = model.predict(processed)
    return int(tf.argmax(prediction, axis=1).numpy()[0])
```

---

### `preprocessing.py`

```python
import cv2
import numpy as np

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)
    return reshaped
```

---

## 사용 기술

- **Python 3**
- **OpenCV** – 실시간 영상 처리
- **TensorFlow/Keras** – 딥러닝 숫자 인식 모델
- **NumPy** – 이미지 배열 처리

---

