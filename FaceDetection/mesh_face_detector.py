"""
Module: mesh_face_detector.py
Author: Jacob Pitsenberger
Date: 10-2-23

Description:
    This module provides a FaceMeshDetector class that handles face detection using the OpenCV and
    Google MediaPipe libraries.
"""
import cv2
import mediapipe as mp
import numpy as np


class FaceMeshDetector:
    """
    FaceMeshDetector class for detecting facial landmarks using Mediapipe.
    """
    FONT = cv2.FONT_HERSHEY_COMPLEX
    TEXT_COLOR = (0, 255, 255)
    BOX_COLOR = (0, 0, 255)
    THICKNESS = 2

    def __init__(self, staticMode=False, maxFaces=10, refine_landmarks=True, minDetectionCon=0.4, minTrackCon=0.5):
        """
        Constructor method to initialize the FaceMeshDetector object.

        Args:
            staticMode (bool): If True, face landmarks are not refined for every frame.
            maxFaces (int): Maximum number of faces to detect.
            refine_landmarks (bool): If True, the face landmarks are refined.
            minDetectionCon (float): Minimum confidence value for a face detection to be considered successful.
            minTrackCon (float): Minimum confidence value for a face to be considered successfully tracked.
        """
        self.results = None  # Store the results of face detection and landmarks
        self.imgRGB = None  # Store the RGB version of the input image
        self.staticMode = staticMode  # Flag for static mode
        self.maxFaces = maxFaces  # Maximum number of faces to detect
        self.refine_landmarks = refine_landmarks  # Flag for refining landmarks
        self.minDetectionCon = minDetectionCon  # Minimum confidence for face detection
        self.minTrackCon = minTrackCon  # Minimum confidence for face tracking

        # Initialize Mediapipe drawing utilities and face mesh model
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.refine_landmarks,
                                                 self.minDetectionCon, self.minTrackCon)

        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def detect_faces(self, frame: np.ndarray) -> None:
        """
        Detects facial landmarks in an image.

        Args:
            frame (numpy.ndarray): Input image (BGR format).

        Returns:
            None
        """
        self.imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR image to RGB
        self.results = self.faceMesh.process(self.imgRGB)  # Process the image with the face mesh model

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                self.drawBoundingBox(frame, faceLms)
            # Update the face count with the number of faces detected
            face_count = "Faces: " + str(len(self.results.multi_face_landmarks))
            cv2.putText(frame, face_count, (10, 25), self.FONT, 1, self.TEXT_COLOR, self.THICKNESS)
            print(face_count)

    def drawBoundingBox(self, frame: np.ndarray, faceLms: mp.solutions.face_mesh.NamedTuple) -> None:
        """
        Draws a bounding box around the detected face mesh.

        Args:
            frame (numpy.ndarray): Input image (BGR format).
            faceLms (typing.NamedTuple): Detected face landmarks.

        Returns:
            None
        """
        # Get bounding box coordinates
        ih, iw, ic = frame.shape
        x_min, x_max, y_min, y_max = iw, 0, ih, 0

        for id, lm in enumerate(faceLms.landmark):
            x, y = int(lm.x * iw), int(lm.y * ih)
            if x < x_min:
                x_min = x
            if x > x_max:
                x_max = x
            if y < y_min:
                y_min = y
            if y > y_max:
                y_max = y

        # Draw the bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), self.BOX_COLOR, self.THICKNESS)
