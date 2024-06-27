from abc import ABC, abstractmethod


class AnalyserInterface(ABC):

    @abstractmethod
    def analyse(self, detection_result):
        pass

    
    @abstractmethod
    def draw_landmarks_on_image(self, rgb_image, detection_result):
        pass


class DetectorBase(ABC):

    def __init__(self) -> None:
        self._detection_results = None
        self._analyser_results = None
        self._frame = None
        self._annotated_image = None


    @abstractmethod
    def analyse_async(self, frame, frame_timestamp_ms) -> None:
        """Asynchronously analyzes a frame and processes the landmarks.
        Args:
            frame: The frame to be analyzed.
            frame_timestamp_ms: The timestamp of the frame in milliseconds.
        """
        pass
    
    @abstractmethod
    def get_result_string(self): 
        """Returns a string representation of the results."""  
        pass


    def process_landmarks(self, analyser: AnalyserInterface):
        self._analyser_results = analyser.analyse(self._detection_results)
        self._annotated_image  = analyser.draw_landmarks_on_image(
            self._frame, self._detection_results)


    def get_annotated_image(self):
        """Returns the annotated image."""
        return self._annotated_image
    
    @abstractmethod
    def close(self):
        """Closes the detector and performs any necessary cleanup operations."""
        pass