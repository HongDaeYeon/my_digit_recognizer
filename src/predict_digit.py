import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils.preprocessing import preprocess_image
import numpy as np

if not hasattr(np, 'bool'):
    np.bool = bool

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model/digit_model"))
model = load_model(model_path)

def predict_digit(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "이미지를 불러올 수 없습니다: " + image_path
    
    processed = preprocess_image(image)

    print("전처리 결과 type:", type(processed))
    print("전처리 결과 shape:", getattr(processed, "shape", "없음"))

    try:
        processed = np.array(processed, dtype=np.float32)
        prediction = model.predict(processed)
        return int(tf.argmax(prediction, axis=1).numpy()[0])
    except Exception as e:
        return f"예측 중 오류 발생: {str(e)}"

if __name__ == "__main__":
    image_path = os.path.join(os.path.dirname(__file__), "../src/8.jpg")
    result = predict_digit(image_path)
    print("예측 결과:", result)