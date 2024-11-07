import cv2
import gesture_analyser
import pose_analyser
import live_stream_gesture_detector as stream_detector
import live_stream_pose_detector as pose_detector

# Create a GestureAnalyser object
analyser_hand = gesture_analyser.GestureAnalyser()
detector_hand = stream_detector.LiveStreamGestureDetector()
analyser_pose = pose_analyser.PoseAnalyser()
detector_pose = pose_detector.LiveStreamPoseDetector()

# Initialize the detector and analyser with the pose model
analyser = analyser_pose
detector = detector_pose

 # Get Webcam video stream with OpenCV
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Analyse the frame asynchronously
    frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    detector.analyse_async(frame, frame_timestamp_ms)
    detector.process_landmarks(analyser)

    print(f"Detected: {detector.get_result_string()}")

    cv2.imshow('MediaPipe Pose Landmarks', detector.get_annotated_image())


    key_pressed = cv2.waitKey(100)
    # check for the 'esc' key to exit
    # check for the '1' key to enable pose detection
    # check for the '2' key to enable hand detection
    # wait for 100ms before next iteration
    if (key_pressed & 0xFF) == 27:
        break
    if (key_pressed & 0xFF) == ord('1'):
        analyser = analyser_pose
        detector = detector_pose
    if (key_pressed & 0xFF) == ord('2'):
        analyser = analyser_hand
        detector = detector_hand

cap.release()
detector_hand.close()
detector_pose.close()
cv2.destroyAllWindows()
