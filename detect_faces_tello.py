"""
Jacob Pitsenberger

This module demonstrates using openCV with the TelloEDU mini drone to perform facial detection
and keep a count of the number of faces detected at a given time.

REVISIONS
1. 11/2/23 - added a variable for the only effect currently functional (face blur). This is used when
             initializing either detector instance and can be set to 'None' for standard detection rectangles.
"""

import cv2
from djitellopy import tello
from FaceDetection.frontal_face_detector import FrontalFaceDetector
from FaceDetection.mesh_face_detector import FaceMeshDetector

def run_tello_video(drone, detector):
    """For use in main with djitellopy tello object"""
    while True:
        frame = drone.get_frame_read().frame
        detector.detect_faces(frame)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def main():
    effect = 'blur'
    haar_detector = FrontalFaceDetector(effects=effect)
    mesh_detector = FaceMeshDetector(effects=effect)
    # Connect to the drone and start receiving video
    drone = tello.Tello()
    drone.connect()
    drone.streamon()
    run_tello_video(drone, detector=mesh_detector)


if __name__ == "__main__":
    main()
