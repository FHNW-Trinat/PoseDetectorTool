import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class LiveStreamPoseDetector:

    def __init__(self) -> None:
        self._detector = self.create_livestream_detector()
        self._detection_results = None
    

    def create_livestream_detector(self) -> vision.PoseLandmarker:
        """Creates and returns a PoseLandmarker instance for live stream mode.
        Returns:
            detector: PoseLandmarker instance configured for live stream mode.
        """
        # callback function to update the detection results
        def update_result(result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            self._detection_results = result

        # Create a pose landmarker instance in the live stream mode:
        base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=update_result)

        return vision.PoseLandmarker.create_from_options(options)
    

    def analyse_async(self, frame, frame_timestamp_ms) -> None:
        """Asynchronously analyzes a frame and processes the pose landmarks.
        Args:
            frame: The frame to be analyzed.
            frame_timestamp_ms: The timestamp of the frame in milliseconds.
        """
        # Convert the OpenCV image to MediaPipe Image.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # Process the image and get the pose landmarks (results processing in callback).
        self._detector.detect_async(mp_image, frame_timestamp_ms)


    def get_detection_results(self) -> vision.PoseLandmarkerResult:
        """Returns the detection results."""
        return self._detection_results
    
    def close(self):
        """Closes the detector and performs any necessary cleanup operations."""
        self._detector.close()