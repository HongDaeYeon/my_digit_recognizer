**my digit recognizer**  

웹캠으로 영상 내의 손글씨 숫자를 인식하는 딥러닝 기반의 숫자 인식기입니다. OpenCV로 실시간 카메라 영상에서 숫자가 포함된 영역을 잘라내고 전처리 및 딥러닝 모델을 통해 예측 결과를 화면에 표시합니다.

---

**개요**

이 프로젝트는 TensorFlow 기반 숫자 분류 모델을 이용해 캡쳐된 영상에서 숫자를 인식합니다. 중심 사각형 영역(ROI)에 손글씨 숫자를 보여주면 실시간으로 예측된 숫자가 화면에 출력됩니다.

---

**실행 방법**

1. 필수 패키지 설치

   ```bash
   pip install opencv-python tensorflow numpy
   ```

2. **실행 명령어**

   ```bash
   python src/digit_recognizer.py
   ```
3. **사용법**
   - 웹캠의 중앙 파란색 사각형 안에 손글씨 숫자가 위치한 영상을 재생
   - 예측된 숫자가 좌측 상단에 실시간으로 표시
   - ESC 키를 누르거나 영상이 끝나면 종료


---

## 주요 기능

- 영상 캡처 인식
- 중심 영역 ROI에서 숫자 영역 추출
- 딥러닝 모델을 통한 숫자 예측
- 예측값 실시간 표시

---

## 디렉터리 구조

```
111/
├── model/
│   └── digit_model         
├── src/
│   ├── train_model.py  
│   ├── predict_digit.py
│   └── digit_recognizer.py       
├── utils/
│       └── preprocessing.py       
└── README.md                    
```

---

## 주요 파일 설명



### `train_model.py`

```
MNIST 손글씨 숫자 이미지 데이터셋을 기반으로 CNN 모델을 학습시키고 이를 저장

데이터 준비	  - MNIST 로드 및 정규화
모델 구성      - CNN
컴파일 및 학습 - 전체 데이터셋 5회 반복학습 
모델 저장	     - model/digit_model 폴더에 저장
```

---

### `predict_digit.py`

```
이미지를 불러와 전처리 후 학습된 모델을 사용해 숫자를 예측

기본 설정 및 모델 불러오기
예측 함수 predict_digit(image_path)
 (__main__)으로 학습된 모델 테스트
```

---

### `digit_recognizer.py`

```
비디오 파일로부터 프레임을 읽고 화면 중앙에서 숫자를 예측

비디오 불러오기
ROI 추출
ROI 이미지로 숫자 예측
예측 결과 시각화
```

---

### `preprocessing.py`

```
일반 이미지를 MNIST 스타일로 전처리해 딥러닝 모델에 입력할 수 있도록 변환
```
---

## 결과 화면

![Image](https://github.com/user-attachments/assets/58d29c57-2b61-49b8-b204-362088ba8651)

![Image](https://github.com/user-attachments/assets/6b40835d-a2a0-49f6-8f05-58cd87129453)

---

## 사용 기술

- **Python 3**
- **OpenCV** – 실시간 영상 처리
- **TensorFlow/Keras** – 딥러닝 숫자 인식 모델
- **NumPy** – 이미지 배열 처리

---

