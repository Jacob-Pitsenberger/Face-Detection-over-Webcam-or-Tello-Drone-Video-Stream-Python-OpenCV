"""
Module: frontal_face_detector.py
Author: Jacob Pitsenberger
Date: 9-27-23

Description:
    This module provides a FrontalFaceDetector class that handles face detection using the OpenCV library.
"""

import cv2
import os
import numpy as np

class FrontalFaceDetector:
    """
    Provides a FrontalFaceDetector class that handles face detection using the OpenCV library.
    """
    FONT = cv2.FONT_HERSHEY_COMPLEX
    TEXT_COLOR = (0, 255, 255)
    BOX_COLOR = (0, 0, 255)
    THICKNESS = 2

    def __init__(self):
        """
        Initialize the FrontalFaceDetector with a pre-trained cascade classifier for face detection.
        """
        # Get the current file directory
        current_dir = os.path.dirname(os.path.realpath(__file__))

        # Construct the path to haarcascade file
        cascade_path = os.path.join(current_dir, 'data-files', 'haarcascade_frontalface_default.xml')

        # Create a cascade
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_faces(self, frame: np.ndarray) -> None:
        """
        Detect faces in a given frame using the pre-trained cascade classifier.

        Args:
            frame: The input frame in which faces will be detected.

        Returns:
            None
        """
        try:
            # Convert the frame to grayscale
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Get numpy array with values for faces detected by passing in grayscale image, scale factor, and minimum neighbors
            faces = self.face_cascade.detectMultiScale(frame_gray, 1.2, 8)

            # For the x, y coordinates and width, height detected
            for (x, y, w, h) in faces:
                # Draw a rectangle around the face using these values
                cv2.rectangle(frame, (x, y), (x + w, y + h), self.BOX_COLOR, self.THICKNESS)

            # Update the face count with the number of faces detected
            face_count = "Faces: " + str(len(faces))
            cv2.putText(frame, face_count, (10, 25), self.FONT, 1, self.TEXT_COLOR, self.THICKNESS)
            print(face_count)

        except Exception as e:
            print(f"Error during face detection: {e}")
