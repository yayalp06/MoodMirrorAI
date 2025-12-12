# MoodMirror AI
**Reflect How You Feel, Uplift How You Live**  
*ITAI 1378 Final Project Proposal 
*by Y. Powell / O, Ogedengbe
 

MoodMirror AI is a smart mirror powered by computer vision that recognizes your facial emotion in real time . It responds with positive affirmations or mood-based music to help boost your motivation and mental well-being.  Built using YOLOv8, PyTorch, and OpenCV, this project blends technology and emotion into a daily self-care experience.

---

##  Overview
MoodMirror AI uses computer vision to recognize a person’s facial emotion in real time and respond with affirmations or music to help boost motivation and emotional well-being.

##  Technical Approach
- YOLOv8 for face detection
- FER2013 CNN for emotion classification
- PyTorch, OpenCV, Streamlit/Gradio
- Optional Spotify API integration
We chose this approach because it combines fast, accurate object detection (YOLOv8) with a proven emotion recognition model (FER2013 CNN) in a lightweight PyTorch framework which is perfect for real time responsiveness and easy deployment.


##  Data Plan
- Dataset: FER2013 (35K+ images)
- Labels: Happy, Sad, Angry, Surprised, Fearful, Disgusted, Neutral
- Source: Kaggle / Roboflow

# Dataset Information

## FER2013 Dataset
- Source: Kaggle / Roboflow
- Type: Facial emotion recognition dataset
- Total Images: ~35,000
- Classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Format: 48x48 grayscale facial images
- License: Public educational use

For best results:
- Resize all images to (224x224)
- Split data 80/10/10 (train/validation/test)
- Apply image augmentation (flips, brightness, rotations)

##  Week-by-Week Plan
Week 10 – Dataset setup  
Week 11 – Model training  
Week 12 – Webcam integration  
Week 13 – Affirmations/music integration  
Week 14 – Testing/documentation  
Week 15 – Presentation
---

## Project Repository Structure (Recommended)

```text
MoodMirror-AI/
├── app.py
├── emotion_model.py
├── affirmations.py
├── AI_usage_log.md
├── README.md
├── requirements.txt
└── notebooks/
    └── 01_exploration.ipynb
```

## Quick Start (Run the Demo)

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Run MoodMirror AI (webcam demo)

```bash
python app.py
```

- Press **Q** to quit.
- The window will show a face box, an emotion label, and a positive affirmation.

> Note: This demo uses a lightweight baseline emotion classifier so it runs immediately without training. The pipeline still matches the intended design: **Camera → Face Detection → Emotion Classification → Affirmation Output**.

---

## Code (Copy into Files)

### `affirmations.py`

```python
AFFIRMATIONS = {
    "Happy": [
        "Keep shining!",
        "Your joy is powerful."
    ],
    "Sad": [
        "It's okay to feel this way.",
        "You are not alone."
    ],
    "Angry": [
        "Pause. Breathe. Reset.",
        "You are in control."
    ],
    "Surprise": [
        "Embrace the moment."
    ],
    "Fear": [
        "You are stronger than you think."
    ],
    "Disgust": [
        "Release what doesn't serve you."
    ],
    "Neutral": [
        "Stay present.",
        "You're doing just fine."
    ]
}
```

### `emotion_model.py` (Baseline demo classifier)

```python
import cv2
import random

EMOTIONS = ["Happy", "Sad", "Angry", "Surprise", "Neutral"]

class EmotionClassifier:
    """Baseline classifier for demo purposes.

    This uses simple image statistics (brightness) + a small randomized set to
    produce a stable, immediate demo without training.

    Swap this class with a trained FER2013 CNN later if desired.
    """

    def predict(self, face_bgr):
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        mean_intensity = float(gray.mean())

        # Simple heuristic buckets (keeps the demo responsive and predictable)
        if mean_intensity > 150:
            return "Happy"
        elif mean_intensity < 90:
            return "Sad"
        else:
            return random.choice(EMOTIONS)
```

### `app.py` (Main application)

```python
import cv2
from ultralytics import YOLO
from emotion_model import EmotionClassifier
from affirmations import AFFIRMATIONS
import random

# Face detection (YOLOv8)
# Using yolov8n for speed; you can swap to yolov8s/yolov8m if desired.
face_model = YOLO("yolov8n.pt")

# Emotion classifier (baseline demo)
emotion_model = EmotionClassifier()

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not access webcam. Check permissions or try a different camera index.")

    print("MoodMirror AI is running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO inference
        results = face_model(frame, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = frame[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                emotion = emotion_model.predict(face)
                affirmation = random.choice(AFFIRMATIONS.get(emotion, AFFIRMATIONS["Neutral"]))

                # Draw detection and text
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, emotion, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
                )
                cv2.putText(
                    frame, affirmation, (30, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
                )

        cv2.imshow("MoodMirror AI", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

---

## AI Usage Log 

# AI Usage Log – MoodMirror AI

Date: Fall 2025

## Tools Used
- ChatGPT (OpenAI)

## How AI Was Used
- Assisted in generating starter Python code for the MoodMirror AI demo, including webcam capture, face detection, emotion labeling logic, and affirmation display.
- Helped organize the project repository structure and format the README documentation clearly and professionally.
- Provided guidance for integrating computer vision components (YOLOv8, OpenCV) into a real-time application pipeline.
- Assisted with troubleshooting runtime issues and refining the demo workflow to ensure the system ran reliably for presentation and video recording.
- Helped draft demo explanations and guidance for recording the final project video.

## Human Contribution
- Conceived the original project idea, goals, and real-world application of MoodMirror AI.
- Selected the datasets, models, and frameworks used in the project based on course material.
- Implemented, tested, and executed the final code locally, including webcam setup and live demonstration.
- Made final decisions regarding system design, simplifications, and demo-ready tradeoffs.
- Reviewed, edited, and validated all AI-assisted content to ensure correctness, clarity, and academic appropriateness.
- Prepared the final submission materials and recorded the demo video.

## Academic Integrity Statement
This project complies with course policies regarding the use of generative AI tools.  
AI-generated content was used strictly as an assistive resource and was reviewed, modified, and integrated by the student.  
All final decisions, testing, and project execution were performed by the student.

---
