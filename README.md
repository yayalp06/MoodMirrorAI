# MoodMirror AI
**Reflect How You Feel, Uplift How You Live**  
*ITAI 1378 Final Project Proposal 
*by Y. Powell / O, Ogedengbe
 
MoodMirror AI is a smart mirror powered by computer vision that recognizes your facial emotion in real time . It responds with positive affirmations or mood-based music to help boost your motivation and mental well-being.  Built using YOLOv8, PyTorch, and OpenCV, this project blends technology and emotion into a daily self-care experience.

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
- # Dataset Information

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
