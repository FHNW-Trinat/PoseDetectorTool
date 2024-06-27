import cv2
import pose_analyser
import live_stream_pose_detector as stream_detector


# Create a PoseAnalyser object
analyser = pose_analyser.PoseAnalyser()
detector = stream_detector.LiveStreamPoseDetector()

 # Get Webcam video stream with OpenCV
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Analyse the frame asynchronously
    frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    detector.analyse_async(frame, frame_timestamp_ms)
    # Get the last detection results from the async analysis
    detection_result = detector.get_detection_results()
        
    # check if detection result is available, if so analyse it
    if detection_result is not None and len(detection_result.pose_landmarks) > 0:
        # Draw the landmarks on the frame
        annotated_image = analyser.draw_landmarks_on_image(frame, detection_result)
          
        # Analyse the detection result for a pose
        detected_pose = analyser.analyse(detection_result)
        print(f"Detected pose: {detected_pose.name}")

    else:
        # Display the original image/frame.
        annotated_image = frame

    cv2.imshow('MediaPipe Pose Landmarks', annotated_image)

    # check for the 'esc' key to exit
    # wait for 100ms before next iteration
    if cv2.waitKey(100) & 0xFF == 27:
        break

cap.release()
detector.close()
cv2.destroyAllWindows()
