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


 # Get Webcam video stream with OpenCV
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Analyse the frame asynchronously
    frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    detector_hand.analyse_async(frame, frame_timestamp_ms)
    detector_hand.process_landmarks(analyser_hand)

    detector_pose.analyse_async(frame, frame_timestamp_ms)
    # Not nice, but: this is to get both hand and pose landmarks into the output image
    detector_pose._frame = detector_hand.get_annotated_image()
    detector_pose.process_landmarks(analyser_pose)

    print(f"Detected pose: {detector_hand.get_result_string()}   {detector_pose.get_result_string()} ")

    cv2.imshow('MediaPipe Pose Landmarks', detector_pose.get_annotated_image())

    # check for the 'esc' key to exit
    # wait for 100ms before next iteration
    if cv2.waitKey(100) & 0xFF == 27:
        break

cap.release()
detector_hand.close()
detector_pose.close()
cv2.destroyAllWindows()
