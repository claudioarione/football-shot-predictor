# Local imports
from pose_estimation.train_classification_model import XGBoostClassifier
from pose_estimation.yolo_model import YOLO
from pose_estimation.poseModule import PoseDetector
from pose_estimation.draw_results import draw_shot_predictions
import utils
# Library imports
import cv2


def analyze_video(path: str, classification_model: XGBoostClassifier, width_rel=20, height_rel=15) -> None:
    # Instantiate the YOLO model for object detection
    detection_model = YOLO()

    # Start the video capture
    video = cv2.VideoCapture(path)

    # Define the size of the final, resized video, in order to avoid exceeding resource limits
    size_factor = 32
    size = (size_factor * width_rel, size_factor * height_rel)

    # Define the pose detectors for attacker and goalkeeper
    attacker_detector, goalkeeper_detector = PoseDetector(), PoseDetector()

    # Define utility variables
    previous_attacker, previous_goalkeeper = None, None
    attacker_features = []
    # Define a variable saying if a stop is needed and another if the prediction has to be computed
    stop = False
    predict = False

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
            stop_time = 100
            if predict:
                # Predict the direction of the shoot
                attacker_data = attacker_features[-33 * 10:]
                lcr_probabilities = classification_model.predict_class(attacker_data)[0]
                # Round the results for visualization purposes
                lcr_probabilities = [format(num * 100, '.1f') for num in lcr_probabilities]
                frame = draw_shot_predictions(frame, lcr_probabilities)

                stop_time = 4000
                predict = False

            cv2.imshow("Image", frame)
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
        landmarks_attacker_list = attacker_detector.get_position(current_attacker)
        attacker_features = utils.preprocess(attacker_features, landmarks_attacker_list)
        utils.process_player(frame, current_goalkeeper, goalkeeper_detector)

        # Draw boundaries for attacker and goalkeeper
        utils.draw_object_bounding_box(frame, current_attacker)
        utils.draw_object_bounding_box(frame, current_goalkeeper)

        cv2.imshow("Image", original_image if stop else frame)

        stop = utils.check_if_stop_video(soccer_ball_box, current_attacker, attacker_detector)
        predict = stop

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()


def show_predictions(video_paths: list, training_path: str):
    # First, train the classification model which will be used later
    classification_model = XGBoostClassifier(training_path)
    classification_model.train_model()

    # Secondly, open videos one by one and predict the outcome
    for path in video_paths:
        analyze_video(path, classification_model)


if __name__ == "__main__":
    # TODO add an argument parser to make users choose if they want to train on a given dataset (whose videos
    #  have to be listed in a file) or to show the prediction on given videos
    video_paths = ['data/Penalty_Neymar.mp4']
    training_data_path = 'data/training_data.npy'
    show_predictions(video_paths, training_data_path)
