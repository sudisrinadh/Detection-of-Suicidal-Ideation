# Detection of Suicidal Ideation Using NLP.

## Overview
This project detects suicidal ideation using anonymous survey responses and behavioral features such as blink count extracted using OpenCV.

## Methodology
- Real-time face detection using OpenCV
- Eye landmark detection and blink count calculation
- Feature extraction from blink behavior
- Machine learning-based risk classification

## Privacy & Ethical Considerations
- No facial images or videos are stored.
- All face processing is performed in real time.
- Blink count is used as a numerical feature only.
- All datasets are anonymized.
- This project is strictly for academic purposes.

## Technologies Used
- Python
- OpenCV
- MediaPipe
- Scikit-learn
- Flask
- NLP

## How to Run
```bash
pip install -r requirements.txt
python app/test1.py
