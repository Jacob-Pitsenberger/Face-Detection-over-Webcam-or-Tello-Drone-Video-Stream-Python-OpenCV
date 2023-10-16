## Face Detection and Count

This repository demonstrates using OpenCV with the TelloEDU mini drone or a computer's web camera to perform facial detection and keep track of the number of faces detected at a given time.

**Author:** Jacob Pitsenberger
**Date:** 10/16/23

### Purpose

This repository showcases how to utilize OpenCV for real-time facial detection using the TelloEDU mini drone's camera or a computer's webcam. It detects faces in the camera stream and draws rectangles around them, along with counting the number of faces detected.

### Instructions

To run the module, ensure you have the required dependencies installed:

```bash
pip install opencv-python
```
Run the script `detect_faces_webcam.py` and observe the facial detection over your computers internal webcam stream. 

Run the script `detect_faces_tello.py` and observe the facial detection over your Tello drones video stream. 

Both scripts utilize the FrontalFaceDetector class from the `frontal_face_detector.py` which utilizes the `haarcascade_frontalface_default.xml` from the `data-files` folder of the `FaceDetection` package to perform facial detections and draw them over our video stream frames.