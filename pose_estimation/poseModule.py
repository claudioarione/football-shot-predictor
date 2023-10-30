import cv2
import mediapipe as mp
import time
from ultralytics import YOLO
import numpy as np


class PoseDetector:
    # TODO: analyze params of Pose class
    def __init__(self, static_image_mode=False, model_complexity=1, smooth_landmarks=True, enable_segmentation=False,
                 smooth_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.results = None
        # self.static_image_mode = static_image_mode
        # self.model_complexity = model_complexity
        # self.smooth_landmarks = smooth_landmarks
        # self.enable_segmentation = enable_segmentation
        # self.smooth_segmentation = smooth_segmentation
        # self.min_detection_confidence = min_detection_confidence
        # self.min_tracking_confidence = min_tracking_confidence
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode, model_complexity, smooth_landmarks,
                                      enable_segmentation, smooth_segmentation, min_detection_confidence,
                                      min_tracking_confidence)

    def find_pose(self, img, draw=True):
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_RGB.flags.writeable = False  # only to improve performance
        self.results = self.pose.process(img_RGB)
        img_RGB.flags.writeable = True
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return img

    def get_position(self, img, draw=True):
        landmarks_list = []
        if self.results.pose_landmarks:
            for id, landmark in enumerate(self.results.pose_landmarks.landmark):
                height, width, channels = img.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                landmarks_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return landmarks_list







# pose_detector = PoseDetector()
#
# # Iterate through the frames and detected bounding boxes
# for frame, detections in zip(results.frames, results.pred):
#     # Filter for only 'person' detections for simplicity
#     person_detections = [det for det in detections if det[4] == "person"]
#
#     # Identify the goalkeeper and the attacker using domain-specific logic
#     goalkeeper_bbox, attacker_bbox = identify_players(person_detections)
#
#     # Get pose for each player
#     goalkeeper_pose_img, goalkeeper_landmarks = get_pose_for_player(goalkeeper_bbox, frame, pose_detector)
#     attacker_pose_img, attacker_landmarks = get_pose_for_player(attacker_bbox, frame, pose_detector)
#

def test():
    # global variables
    # pose_estimator = []
    # pose_estimator_dim = []
    # # For each object detected
    # # WHICH POSE ESTIMATOR TO USE.
    # selected_pose_idx = 0

    cap = cv2.VideoCapture('../data/Penalty_Neymar.mp4')
    previous_time = 0
    detector = PoseDetector()
    while True:  # TODO: maybe is better cap.isOpened() and cap.release()
        success, img = cap.read()
        img = detector.find_pose(img)
        landmarks_list = detector.get_position(img)
        print(landmarks_list)  # debug
        # landmarks_list = detector.get_position(img, draw=False)
        # cv2.circle(img, (landmarks_list[32][1], landmarks_list[32][2]), 20, (0, 0, 255), cv2.FILLED)
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(10)


if __name__ == '__main__':
    test()