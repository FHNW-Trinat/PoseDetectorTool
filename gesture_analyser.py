import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import enum
import numpy
from typing import Tuple
from analyser_interface import AnalyserInterface


class GestureAnalyser(AnalyserInterface):


    def __init__(self):
        pass


    def analyse(self, detection_result):
        """ Analyzes the given detection result and returns the first / most prominent gesture.
        """
        if detection_result is None or len(detection_result.hand_landmarks) == 0:
            return None  
        # Get the landmarks of the first hand 
        return detection_result.gestures[0][0]
    

    def draw_landmarks_on_image(self, rgb_image, detection_result) -> numpy.ndarray:
        """
        Draws the hand landmarks on the given RGB image.
        Args:
            rgb_image (numpy.ndarray): The RGB image on which to draw the landmarks.
            detection_result (HandLandmarkerResult): The detection result containing the pose landmarks.
        Returns:
            numpy.ndarray: The annotated image with the hand landmarks drawn.
        """
        if detection_result is None or len(detection_result.hand_landmarks) == 0:
            return rgb_image # return the original image if no landmarks are detected

        hand_landmarks_list = detection_result.hand_landmarks
        annotated_image = rgb_image.copy()

        # Loop through the detected poses to visualize.
        for idx in range(len(hand_landmarks_list)):
            pose_landmarks = hand_landmarks_list[idx]

            # Draw the pose landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style()
            )
        return annotated_image

