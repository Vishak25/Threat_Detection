import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from model import MILScoringHead
import argparse

YOLO_MODEL = 'yolov8n.pt'
INPUT_SIZE = (224, 224)


def build_feature_extractor():
    """Builds ResNet50V2 feature extractor matching training setup."""
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights='imagenet',
        pooling='avg',
        input_shape=(224, 224, 3)
    )
    return base_model


def preprocess_crop(crop):
    """Preprocesses a crop for ResNet50V2."""
    crop = cv2.resize(crop, INPUT_SIZE)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = tf.keras.applications.resnet_v2.preprocess_input(crop)
    return np.expand_dims(crop, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Real-Time Anomaly Detection")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained MIL weights (.h5)")
    parser.add_argument("--threshold", type=float, default=0.4, help="Anomaly threshold")
    args = parser.parse_args()

    print("Loading YOLOv8...")
    yolo = YOLO(YOLO_MODEL)

    print("Loading Feature Extractor (ResNet50V2)...")
    feature_extractor = build_feature_extractor()

    print(f"Loading MIL Model from {args.weights}...")
    mil_head = MILScoringHead()
    mil_head.build((None, 2048)) 
    mil_head.load_weights(args.weights)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.video}")
        return

    print("Starting Inference... Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo(frame, classes=[0], verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                h, w, _ = frame.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue

                person_crop = frame[y1:y2, x1:x2]

                try:
                    input_tensor = preprocess_crop(person_crop)
                    features = feature_extractor(input_tensor, training=False)
                    score = float(mil_head(features, training=False)[0][0])
                except Exception as e:
                    print(f"Error processing crop: {e}")
                    score = 0.0

                color = (0, 255, 0)
                label = f"Normal: {score:.2f}"

                if score > args.threshold:
                    color = (0, 0, 255)
                    label = f"ANOMALY: {score:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Real-Time Anomaly Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
