"""
Jacob Pitsenberger

This module demonstrates using openCV to perform facial detection and keep a count of the number
of faces detected at a given time in a realtime webcam stream.
"""

import cv2
from FaceDetection.frontal_face_detector import FrontalFaceDetector


def detect_over_webcam(cap, detector):
    """For use in main with openCV capture object"""
    while True:
        ret, frame = cap.read()
        detector.detect_faces(frame)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    haar_detector = FrontalFaceDetector()
    # Create capture object for computers camera
    cap = cv2.VideoCapture(0)
    detect_over_webcam(cap, detector=haar_detector)


if __name__ == "__main__":
    main()
