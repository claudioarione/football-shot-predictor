from pose_estimation.yolo_model import YOLO
from pose_estimation.poseModule import PoseDetector
import utils
import cv2
import numpy as np
import pandas as pd


def analyze_video(path: str, output: int, width_rel=30, height_rel=22) -> tuple:
    model = YOLO()

    video = cv2.VideoCapture(path)
    size_factor = 32

    size = (size_factor * width_rel, size_factor * height_rel)

    attacker_detector, goalkeeper_detector = PoseDetector(), PoseDetector()
    previous_attacker, previous_goalkeeper = None, None
    attacker_features = []
    goalkeeper_features = []
    # Define a variable saying if a stop is needed and another if the prediction has to be computed
    stop = False

    # Defining loop for catching frames
    while True:
        ret, frame = video.read()
        if not ret:
            break
        # Resize the frame
        frame = cv2.resize(frame, size)
        original_image = frame.copy()

        if stop:
            break

        outputs = model.evaluate(frame)
        boxes, class_scores, soccer_ball_box = utils.identify_all_objects(outputs, size)
        if soccer_ball_box:
            utils.draw_object_bounding_box(frame, soccer_ball_box)

        # Selection
        filtered_boxes = utils.non_maxima_suppression(boxes, class_scores)

        # Identify attacker and goalkeeper
        current_attacker, current_goalkeeper = utils.identify_players(
            filtered_boxes, previous_attacker, previous_goalkeeper
        )
        previous_attacker, previous_goalkeeper = current_attacker, current_goalkeeper

        # Process players (pose estimation)
        utils.process_player(frame, current_attacker, attacker_detector)
        utils.process_player(frame, current_goalkeeper, goalkeeper_detector)
        landmarks_attacker_list = attacker_detector.get_position(current_attacker)
        landmarks_goalkeeper_list = attacker_detector.get_position(current_goalkeeper)
        attacker_features = utils.preprocess(attacker_features, landmarks_attacker_list)
        goalkeeper_features = utils.preprocess(
            goalkeeper_features, landmarks_goalkeeper_list
        )

        # Draw boundaries for attacker and goalkeeper
        utils.draw_object_bounding_box(frame, current_attacker)
        utils.draw_object_bounding_box(frame, current_goalkeeper)

        cv2.imshow("Football Shot Predictor", frame)

        stop = utils.check_if_stop_video(
            soccer_ball_box, current_attacker, attacker_detector
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()
    att_label, gk_label = output
    # Select 33 keypoints * last 10 frames before kick
    return np.append(attacker_features[-33 * 10 :], att_label), np.append(
        goalkeeper_features[-33 * 5 :], gk_label
    )


# FIXME: this is for training
def create_training_dataset(
    training_dataframe: pd.DataFrame, save_training_data_path: str
):
    att_dataset, gk_dataset = [], []
    videos = dict(
        zip(
            training_dataframe["link"],
            zip(training_dataframe["att_label"], training_dataframe["gk_label"]),
        )
    )
    print(videos)
    for video_path, label in videos.items():
        attacker_data, goalkeeper_data = analyze_video(path=video_path, output=label)
        att_dataset.append(attacker_data)
        gk_dataset.append(goalkeeper_data)

    # Save the training data of both attacker and goalkeeper
    att_dataset_array = np.array(att_dataset)
    gk_dataset_array = np.array(gk_dataset)
    np.save(save_training_data_path + "/att_training_data.npy", att_dataset_array)
    np.save(save_training_data_path + "/gk_training_data.npy", gk_dataset_array)
