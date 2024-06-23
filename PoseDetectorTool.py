import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import pose_analyser


# Create a PoseAnalyser object
analyser = pose_analyser.PoseAnalyser()

detected_pose, annotated_image = analyser.analyse_image("Doku/pose.jpg")

print(f"Detected pose: {detected_pose.name}")

# Convert the mediapipe image to OpenCV format and display it
cv2.imshow("Output", cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
cv2.destroyAllWindows()