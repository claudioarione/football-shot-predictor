from pose_estimation.yolo_model import YOLO
from pose_estimation.poseModule import PoseDetector
import utils
import cv2
import numpy as np


def analyze_video(path: str, output: int, data: list, width_rel=20, height_rel=15) -> list:
    model = YOLO()

    video = cv2.VideoCapture(path)
    size_factor = 32

    size = (size_factor * width_rel, size_factor * height_rel)

    attacker_detector, goalkeeper_detector = PoseDetector(), PoseDetector()
    previous_attacker, previous_goalkeeper = None, None
    attacker_features = []
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
        landmarks_attacker_list = attacker_detector.get_position(current_attacker)
        attacker_features = utils.preprocess(attacker_features, landmarks_attacker_list)
        utils.process_player(frame, current_goalkeeper, goalkeeper_detector)

        # Draw boundaries for attacker and goalkeeper
        utils.draw_object_bounding_box(frame, current_attacker)
        utils.draw_object_bounding_box(frame, current_goalkeeper)

        cv2.imshow("Image", frame)

        stop = utils.check_if_stop_video(soccer_ball_box, current_attacker, attacker_detector)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()
    data.append(
        np.append(attacker_features[-33 * 10:], output)  # FIXME: this is for training
    )  # 33 keypoints * last 10 frames before kick

    return data


# FIXME: this is for training
def create_training_dataset():
    dataset = []
    videos = {
        "data/Penalty_Neymar.mp4": 2,
        "data/Penalty_Lampard.mp4": 1,
        "data/Penalty_Mata.mp4": 2,
        "data/Penalty_Olic.mp4": 0 # FIXME this label is incorrect but the data must have at least one 0, 1 and 2
    }
    for video_path, label in videos.items():
        dataset = analyze_video(path=video_path, output=label, data=dataset)
    dataset_array = np.array(dataset)
    np.save("data/training_data.npy", dataset_array)


if __name__ == "__main__":
    create_training_dataset()
