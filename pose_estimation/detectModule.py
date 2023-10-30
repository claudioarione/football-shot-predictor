# import cv2
# import mediapipe as mp
# import time
from ultralytics import YOLO
# import numpy as np
#
#
# def objects_detection(original_image):
#     model = YOLO('yolov8n.pt')
#     # The following operations returns a list of Result objects - one each video frame!
#
#     frames = model(source="../data/Penalty_Neymar.mp4", show=True, conf=0.4, save=True, stream=True)
#     for frame in frames:
#         print("frame:", frame)
#         goalkeeper, attacker = identify_players(frame, model.names)
#         annotated_img = annotate_frame(original_image, goalkeeper, attacker)
#         cv2.imshow('Detected Video', annotated_img)
#     cv2.destroyAllWindows()
#
#
# def annotate_frame(original_image, goalkeeper, attacker):
#     # Draw rectangles and labels for the goalkeeper and attacker
#     for box, label in zip([goalkeeper, attacker], ['Goalkeeper', 'Attacker']):
#         x, y, w, h = map(int, box[0])
#         cv2.rectangle(original_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(original_image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#     return original_image
#
#
# def identify_players(detections, names):
#     # For simplicity, only consider the two largest detected persons, assuming they are the goalkeeper and the attacker
#     boxes = [d.boxes.xywh for d in detections if names[int(d.boxes.cls)] == 'person']
#     sorted_boxes = sorted(boxes, key=lambda x: x[0][2] * x[0][3], reverse=True)[:2]
#     # The player closer to the center of the goal (typically upper half of the image) is likely the goalkeeper
#     goalkeeper = min(sorted_boxes, key=lambda x: x[0][1])
#     attacker = max(sorted_boxes, key=lambda x: x[0][1])
#
#     return goalkeeper, attacker
#
#
# if __name__ == '__main__':
#     cap = cv2.VideoCapture('../data/Penalty_Neymar.mp4')
#     while True:
#         success, img = cap.read()
#         if not success:
#             break
#         objects_detection(img)
#     # objects_detection()
#     cap.release()
#     cv2.destroyAllWindows()


if __name__ == '__main__':
    model = YOLO('yolov8m-pose.pt')
    results = model(source='../data/Penalty_Neymar.mp4', show=True, conf=0.4)
