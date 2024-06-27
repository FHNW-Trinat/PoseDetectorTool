import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class LiveStreamGuestureDetector:

    def __init__(self) -> None:
        self._detector = self.create_livestream_detector()
        self._detection_results = None
    

    def create_livestream_detector(self) -> vision.HandLandmarker:
        """Creates and returns a HandLandmarker instance for live stream gesture detection.
        Returns:
            detector: HandLandmarker instance configured for live stream gesture detection.
        """
        # callback function to update the detection results
        def update_result(result: vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
            self._detection_results = result

        # Create a pose landmarker instance in the live stream mode:
        base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=update_result)

        return vision.GestureRecognizer.create_from_options(options)
    

    def analyse_async(self, frame, frame_timestamp_ms) -> None:
        """Asynchronously analyzes a frame and processes the gesture landmarks.
        Args:
            frame: The frame to be analyzed.
            frame_timestamp_ms: The timestamp of the frame in milliseconds.
        """
        # Convert the OpenCV image to MediaPipe Image.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # Process the image and get the pose landmarks (results processing in callback).
        self._detector.recognize_async(mp_image, frame_timestamp_ms)


    def get_detection_results(self) -> vision.HandLandmarkerResult:
        """Returns the detection results."""
        return self._detection_results
    
    def close(self):
        """Closes the detector and performs any necessary cleanup operations."""
        self._detector.close()