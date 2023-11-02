"""
Jacob Pitsenberger

This module demonstrates using openCV to perform facial detection and keep a count of the number
of faces detected at a given time in a realtime webcam stream.

REVISIONS
1. 11/2/23 - added a variable for the only effect currently functional (face blur). This is used when
             initializing either detector instance and can be set to 'None' for standard detection rectangles.
"""

import cv2
from FaceDetection.frontal_face_detector import FrontalFaceDetector
from FaceDetection.mesh_face_detector import FaceMeshDetector


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
    effect = 'blur'
    haar_detector = FrontalFaceDetector(effects=effect)
    mesh_detector = FaceMeshDetector(effects=effect)

    # Create capture object for computers camera
    cap = cv2.VideoCapture(0)

    detect_over_webcam(cap, detector=mesh_detector)


if __name__ == "__main__":
    main()
