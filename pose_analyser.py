import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import enum
from typing import Tuple


class PoseDetected(enum.IntEnum):
    """The poses that can be detected."""
    NO_DETECTION = 0
    RIGHT_HAND_UP = 1
    LEFT_HAND_UP = 2
    BOTH_HANDS_UP = 3




class PoseAnalyser:

    # class attribute
    handup_threshold = 0.2 # minimum distance in meters between hand and nose to be considered a hand up

    def __init__(self):
        pass


    def analyse(self, detection_result) -> PoseDetected:
        """Analyses the landmarks for known poses and returns the first detected one
       (-y)
        ^
        |   R       -0.6
        |     N     -0.5
        |
        |         L -0.2
        |
        0 ---------> +x (0 is body's center of mass / hips )
        |
        +y
        :param detection_result: landmarks calculated from mediapipe 
        :return: PoseDetected enum
        """        
        landmarks = detection_result.pose_world_landmarks[0]
        PoseLandmark = solutions.pose.PoseLandmark
        y_pos = { 'RIGHT_WRIST': landmarks[PoseLandmark.RIGHT_WRIST].y,
                  'LEFT_WRIST': landmarks[PoseLandmark.LEFT_WRIST].y,
                  'NOSE': landmarks[PoseLandmark.NOSE].y
        }

        # Check if right hand is above nose and left hand below.
        if (y_pos['RIGHT_WRIST'] + self.handup_threshold) < y_pos['NOSE'] and \
           (y_pos['LEFT_WRIST']  - self.handup_threshold) > y_pos['NOSE']:
            detected_pose = PoseDetected.RIGHT_HAND_UP
        # Check if left hand is above nose and right hand below.
        elif (y_pos['RIGHT_WRIST'] - self.handup_threshold) > y_pos['NOSE'] and \
             (y_pos['LEFT_WRIST']  + self.handup_threshold) < y_pos['NOSE']:
            detected_pose = PoseDetected.LEFT_HAND_UP
        # Check if both hands are above nose.
        elif (y_pos['RIGHT_WRIST'] + self.handup_threshold/2) < y_pos['NOSE'] and \
             (y_pos['LEFT_WRIST']  + self.handup_threshold/2) < y_pos['NOSE']:
            detected_pose = PoseDetected.BOTH_HANDS_UP
        else:
            detected_pose = PoseDetected.NO_DETECTION

        return detected_pose
    

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        """
        Draws the pose landmarks on the given RGB image.
        Args:
            rgb_image (numpy.ndarray): The RGB image on which to draw the landmarks.
            detection_result (PoseLandmarkerResult): The detection result containing the pose landmarks.
        Returns:
            numpy.ndarray: The annotated image with the pose landmarks drawn.
        """
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = rgb_image.copy()

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style()
            )
        return annotated_image


    def detect_pose(self, image: str) -> Tuple[vision.PoseLandmarkerResult, mp.Image]:
        """
        Detects pose landmarks from an input image.
        param: image (str): The path to the input image.
        return: Tuple[vision.PoseLandmarkerResult, mp.Image]: detection result and the input image.
        """
        # STEP 2: Create a PoseLandmarker object.
        base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True)

        detector = vision.PoseLandmarker.create_from_options(options)

        # STEP 3: Load the input image.
        mp_image = mp.Image.create_from_file(image)

        # STEP 4: Detect pose landmarks from the input image.
        detection_result = detector.detect(mp_image)

        return detection_result, mp_image
    

    def analyse_image(self, image : str, create_annotated_image=True) -> Tuple[PoseDetected, mp.Image]:
        """Analyse the pose in the image and return the detected pose
        :param image: image file path
        :return: Tuple[PoseDetected enum, mp.Image]
        """
        detection_result, mp_image = self.detect_pose(image)
    
        # STEP 5: Process the detection result. In this case, visualize it.
        detected_pose = self.analyse(detection_result)

        annotated_image = None
        if create_annotated_image:
            annotated_image = self.draw_landmarks_on_image(mp_image.numpy_view(), detection_result)

        return detected_pose, annotated_image
    