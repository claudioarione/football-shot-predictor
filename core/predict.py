from core.classification_model import XGBoostClassifier
from pose_estimation.yolo_model import YOLO
from pose_estimation.pose_module import PoseDetector
from utils.draw_results import draw_shot_predictions, draw_dive_prediction
from utils import utils

import cv2


def analyze_video(
    path: str,
    att_classification_model: XGBoostClassifier,
    gk_classification_model: XGBoostClassifier,
    width_rel: int = 30,
    height_rel: int = 22,
    save_video_path: str = None,
) -> None:
    """
    Analyzes a football match video to predict the direction of shots and goalkeeper dives.

    This function processes each frame of the video using the YOLO model for object detection and PoseDetector for pose estimation.
    It identifies the attacker and goalkeeper, tracks their movements, and uses pre-trained classification models to predict the
    outcome of shots and dives.

    :param path: The file path of the video to be analyzed.
    :param att_classification_model: The pre-trained XGBoost classifier model for predicting the attacker's shot direction.
    :param gk_classification_model: The pre-trained XGBoost classifier model for predicting the goalkeeper's dive direction.
    :param width_rel: Relative width for resizing video frames for processing. Default is 30.
    :param height_rel: Relative height for resizing video frames for processing. Default is 22.
    """
    detection_model = YOLO()

    video = cv2.VideoCapture(path)
    fps = video.get(cv2.CAP_PROP_FPS)
    size_factor = 32
    size = (size_factor * width_rel, size_factor * height_rel)
    slow_motion_part = int(2.5 * fps)
    # Define the pose detectors for attacker and goalkeeper
    attacker_detector, goalkeeper_detector = PoseDetector(), PoseDetector()

    # Define utility variables
    previous_attacker, previous_goalkeeper = None, None
    attacker_features = []
    goalkeeper_features = []
    # Define a variable saying if a stop is needed and another if the prediction has to be computed
    stop = False
    predict = False
    # Create a VideoWriter object to save the video
    video_writer = None
    if save_video_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")  # or use 'MP4V' for .mp4 format
        video_writer = cv2.VideoWriter(save_video_path, fourcc, fps, size)

    # Defining loop for catching frames
    while True:
        ret, frame = video.read()
        if not ret:
            break
        # Resize the frame
        frame = cv2.resize(frame, size)
        original_image = frame.copy()

        if stop:
            # The bounding boxes and the pose estimators does not have to be drawn
            stop_time = 50
            if predict:
                # Predict the direction of the shoot
                attacker_data = attacker_features[-33 * 10 :]
                att_lcr_probabilities = att_classification_model.predict_class(
                    attacker_data
                )[0]
                # Round the results for visualization purposes
                att_lcr_probabilities = [
                    format(num * 100, ".1f") for num in att_lcr_probabilities
                ]
                frame = draw_shot_predictions(frame, att_lcr_probabilities)
                for _ in range(slow_motion_part):
                    if video_writer is not None:
                        video_writer.write(frame)
                cv2.imshow("Football Shot Predictor", frame)
                cv2.waitKey(2500)

                # Reset the frame to the original one
                frame = original_image.copy()

                # Predict the direction of the goalkeeper's dive
                goalkeeper_data = goalkeeper_features[-33 * 5 :]
                gk_lr_probabilities = gk_classification_model.predict_class(
                    goalkeeper_data
                )[0]
                gk_lr_probabilities = [
                    format(num * 100, ".1f") for num in gk_lr_probabilities
                ]
                frame = draw_dive_prediction(
                    frame, gk_lr_probabilities, previous_goalkeeper
                )
                for _ in range(slow_motion_part):
                    if video_writer is not None:
                        video_writer.write(frame)

                stop_time = 2500
                predict = False
            if video_writer is not None:
                video_writer.write(frame)
            cv2.imshow("Football Shot Predictor", frame)
            cv2.waitKey(stop_time)
            continue

        # Identify the ball and the candidate boxes of attacker and goalkeeper
        outputs = detection_model.evaluate(frame)
        boxes, class_scores, soccer_ball_box = utils.identify_all_objects(outputs, size)

        # Draw the box correspondant to the ball
        if soccer_ball_box:
            utils.draw_object_bounding_box(frame, soccer_ball_box)

        # Filter the boxes according to the class score and identify attacker and goalkeeper
        filtered_boxes = utils.non_maxima_suppression(boxes, class_scores)
        current_attacker, current_goalkeeper = utils.identify_players(
            filtered_boxes, previous_attacker, previous_goalkeeper
        )
        previous_attacker, previous_goalkeeper = current_attacker, current_goalkeeper

        # Process players (pose estimation)
        utils.process_player(frame, current_attacker, attacker_detector)
        utils.process_player(frame, current_goalkeeper, goalkeeper_detector)
        landmarks_attacker_list = attacker_detector.get_position(current_attacker)
        landmarks_goalkeeper_list = goalkeeper_detector.get_position(current_goalkeeper)
        attacker_features = utils.preprocess(attacker_features, landmarks_attacker_list)
        goalkeeper_features = utils.preprocess(
            goalkeeper_features, landmarks_goalkeeper_list
        )

        # Draw boundaries for attacker and goalkeeper
        utils.draw_object_bounding_box(frame, current_attacker)
        utils.draw_object_bounding_box(frame, current_goalkeeper)

        if video_writer is not None:
            video_writer.write(original_image if stop else frame)
        cv2.imshow("Football Shot Predictor", original_image if stop else frame)

        stop = utils.check_if_stop_video(
            soccer_ball_box, current_attacker, attacker_detector
        )
        predict = stop

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    if video_writer is not None:
        video_writer.release()
    video.release()
    cv2.destroyAllWindows()


def show_predictions(video_paths: list, training_path: str):
    """
    Processes a list of video paths to predict football shot outcomes.

    For each video in the provided list, this function uses the `analyze_video` function to predict the outcome of shots.
    It involves training XGBoost classification models using the provided training data, and then applying these models to
    analyze the videos.

    :param video_paths: A list of file paths for the videos to be analyzed.
    :param training_path: The file path where the training data for the models is stored.
    """
    att_data_path = training_path + "/att_training_data.npy"
    gk_data_path = training_path + "/gk_training_data.npy"

    att_classification_model = XGBoostClassifier(att_data_path, 3)
    att_classification_model.train_model()

    gk_classification_model = XGBoostClassifier(gk_data_path, 2)
    gk_classification_model.train_model()

    # Secondly, open videos one by one and predict the outcome
    for path in video_paths:
        analyze_video(
            path,
            att_classification_model,
            gk_classification_model,
            # save_video_path=path[:-4] + "_predicted.mp4",
        )
