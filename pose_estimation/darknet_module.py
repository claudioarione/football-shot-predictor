import cv2
import numpy as np
import poseModule as pm
from pose_estimation.yolo_model import YOLO
import cvzone
import cvzone.ColorModule as cm
import draw_results as draw
from train_classification_model import XGBoostClassifier


def find_ball_bounding_box(frame, color_finder: cm.ColorFinder, hsv_vals: dict):
    _, mask = color_finder.update(frame, hsv_vals)
    _, contours = cvzone.findContours(frame, mask, minArea=1000)
    return contours[0]["bbox"] if contours else None


def draw_object_bounding_box(image_to_process, box):
    """
    Draws a bounding box around the object in the image.
    :param image_to_process: original image
    :param box: coordinates of the area around the object
    :return: image with marked objects
    """
    if not box:
        return
    x, y, width, height = box
    start_point = (x, y)
    end_point = (x + width, y + height)
    color = (0, 255, 0)
    thickness = 1
    image_with_box = cv2.rectangle(
        image_to_process, start_point, end_point, color, thickness
    )

    return image_with_box


def center_of_box(box) -> np.array:
    """
    :param box: coordinates of the area around the object
    :return: coordinates of the center of the box
    """
    x, y, width, height = box
    return np.array([x + width / 2, y + height / 2])


def euclidean_distance(center1, center2):
    return np.linalg.norm(center1 - center2)


def find_similar_boxes(
        current_frame_boxes, previous_attacker, previous_goalkeeper, threshold: int = 40
):
    # if not previous_large_boxes:
    #     return current_frame_boxes
    # if all(element is None for element in previous_large_boxes):
    #     return current_frame_boxes
    # similar_boxes = []
    # for previous_box in previous_large_boxes:
    #     if previous_box:
    #         prev_center = center_of_box(previous_box)
    #         closest_box = None
    #         closest_distance = float('inf')
    #         for current_box in current_frame_boxes:
    #             current_center = center_of_box(current_box)
    #             distance = euclidean_distance(prev_center, current_center)
    #             if distance < closest_distance:
    #                 closest_distance = distance
    #                 closest_box = current_box
    #         if closest_distance <= threshold:
    #             similar_boxes.append(closest_box)

    # TODO: improve this code
    if not previous_attacker and not previous_goalkeeper:
        return current_frame_boxes

    similar_boxes = []
    if previous_attacker:
        prev_center = center_of_box(previous_attacker)
        # print("attacker_center: ", prev_center)
        closest_box = None
        closest_distance = float("inf")
        for curr_box in current_frame_boxes:
            curr_center = center_of_box(curr_box)
            distance = euclidean_distance(prev_center, curr_center)
            if distance < closest_distance:
                closest_distance = distance
                closest_box = curr_box
        if closest_distance <= threshold:
            similar_boxes.append(closest_box)

    if previous_goalkeeper:
        prev_center = center_of_box(previous_goalkeeper)
        # print("goalkeeper_center: ", prev_center)
        closest_box = None
        closest_distance = float("inf")
        for curr_box in current_frame_boxes:
            curr_center = center_of_box(curr_box)
            distance = euclidean_distance(prev_center, curr_center)
            if distance < closest_distance:
                closest_distance = distance
                closest_box = curr_box
        if closest_distance <= threshold:
            similar_boxes.append(closest_box)
    return similar_boxes


def identify_players(boxes, previous_attacker, previous_goalkeeper):
    """
    For simplicity, only consider the two largest detected persons, assuming they are the goalkeeper and the attacker

    :param boxes: A list of tuples where each one contains the coordinates
                            of the upper-left corner, width, and height of a detected person's
                            bounding box in the format (x, y, width, height).
    :param previous_attacker: The bounding box of the attacker in the previous frame.
    :param previous_goalkeeper: The bounding box of the goalkeeper in the previous frame.
    :return: A tuple containing the bounding boxes of the goalkeeper and attacker,
           or None for each if they cannot be determined.
    """
    if not boxes:
        return None, None

    similar_boxes = find_similar_boxes(boxes, previous_attacker, previous_goalkeeper)

    if not similar_boxes:
        return None, None

    sorted_boxes = sorted(similar_boxes, key=lambda x: x[2] * x[3], reverse=True)

    if len(sorted_boxes) == 1:
        return sorted_boxes[0], None

    goalkeeper = sorted_boxes[1]
    attacker = sorted_boxes[0]
    return attacker, goalkeeper


def non_maxima_suppression(boxes, class_scores):
    chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.4)
    boxes = [boxes[i] for i in chosen_boxes]
    return boxes


def identify_all_objects(outputs, shape, threshold=0):
    """
    :param outputs: 2D NumPy array derived from a CNN
    :param shape: a tuple having the width as first coordinate and the height as second
    :param threshold: the minimum value of the classification score to be accepted
    :return: a tuple consisting of an array containing the coordinates x, y, width and height of all
             the identified human beings and an array of the respective confidence score
    """
    width, height = shape
    class_scores, boxes = [], []
    soccer_ball = None

    for out in outputs:
        for obj in out:
            scores = obj[5:]
            class_index = np.argmax(scores)
            class_score = scores[class_index]

            if class_index != 0 and class_index != 32:
                # The identified object is neither a person nor a ball
                continue

            if class_score > threshold:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                obj_width = int(obj[2] * width)
                obj_height = int(obj[3] * height)

                box = [
                    center_x - obj_width // 2,
                    center_y - obj_height // 2,
                    obj_width,
                    obj_height,
                ]
                if class_index == 0:
                    boxes.append(box)
                    class_scores.append(float(class_score))
                else:
                    soccer_ball = box

    return boxes, class_scores, soccer_ball


def process_player(frame, player, detector: pm.PoseDetector) -> None:
    """
    :param frame: the frame of the video
    :param player: the player to estimate the pose
    :param detector: the mediapipe pose detector
    """
    if not player:
        return
    x, y, w, h = player
    player_image = frame[y: y + h, x: x + w]
    detector.find_pose(player_image)


def check_if_stop_video(
        ball_box, image, attacker_detector: pm.PoseDetector, threshold_in_pixels=25
):
    """
    States if the video must be stopped, i.e., if the foot of the attacker
    is close to impact the ball
    :param ball_box: coordinates of the area around the ball
    :param image: the box of the attacker, represented as an array containing, in order, x, y, width and height
    :param attacker_detector: detector of the attacker
    :param threshold_in_pixels: defines an acceptability threshold for keypoints' proximity to ball
    :return: boolean value
    """

    if ball_box is None:
        return False

    keypoints = attacker_detector.get_position(image)
    # Remember that feet keypoints have an id greater or equal than 27
    feet_keypoints = [keypoint for keypoint in keypoints if keypoint[0] >= 27]

    for keypoint in feet_keypoints:
        keypoint_center = keypoint[1:]
        ball_center = center_of_box(ball_box)
        # Check if the keypoint is enough close to the ball
        if euclidean_distance(keypoint_center, ball_center) < threshold_in_pixels:
            # If so, the video must be stopped
            return True

    # If no keypoint is enough close to the ball, the video should resume normally
    return False


# FIXME: this is for training
def preprocess(all_features: np.array, landmarks_list: list) -> np.array:
    landmarks_array = np.array(landmarks_list[:][:])  # take only feet keypoints
    frame_features = landmarks_array[:, 1:].flatten()  # Exclude keypoint_id
    all_features = np.concatenate([all_features, frame_features])
    return all_features


def analyze_video(path: str, output: int, data: list) -> list:
    model = YOLO()

    video = cv2.VideoCapture(path)
    size_factor = 32
    width_rel, height_rel = 20, 15
    size = (size_factor * width_rel, size_factor * height_rel)

    attacker_detector, goalkeeper_detector = pm.PoseDetector(), pm.PoseDetector()
    previous_attacker, previous_goalkeeper = None, None
    attacker_features = []
    # Define a variable saying if a stop is needed and another if the prediction has to be computed
    stop = False
    predict = False
    debug = False  # TODO change

    # Defining loop for catching frames
    while True:
        ret, frame = video.read()
        if not ret:
            break
        # Resize the frame
        frame = cv2.resize(frame, size)
        original_image = frame.copy()

        if stop:
            # Predict the direction of the shoot if needed
            if predict and not debug:
                # TODO add a global list tailored for classification of this specific video!
                lcr_probabilities = [0.4, 0.4, 0.2]
                cv2.imshow("Image", draw.draw_shot_predictions(frame, lcr_probabilities))
                cv2.waitKey(1000)
                predict = False
            else:
                cv2.imshow("Image", frame)
                cv2.waitKey(100)

            continue

        outputs = model.evaluate(frame)
        boxes, class_scores, soccer_ball_box = identify_all_objects(outputs, size)

        if soccer_ball_box:
            draw_object_bounding_box(frame, soccer_ball_box)

        # Selection
        filtered_boxes = non_maxima_suppression(boxes, class_scores)

        # Identify attacker and goalkeeper
        current_attacker, current_goalkeeper = identify_players(
            filtered_boxes, previous_attacker, previous_goalkeeper
        )
        previous_attacker, previous_goalkeeper = current_attacker, current_goalkeeper

        # Process players (pose estimation)
        process_player(frame, current_attacker, attacker_detector)
        landmarks_attacker_list = attacker_detector.get_position(current_attacker)
        attacker_features = preprocess(attacker_features, landmarks_attacker_list)
        process_player(frame, current_goalkeeper, goalkeeper_detector)

        # Draw boundaries for attacker and goalkeeper
        draw_object_bounding_box(frame, current_attacker)
        draw_object_bounding_box(frame, current_goalkeeper)

        cv2.imshow("Image", original_image if stop else frame)
        wait_until_next_frame = 5000 if stop else 1

        stop = check_if_stop_video(soccer_ball_box, current_attacker, attacker_detector)
        predict = stop

        if cv2.waitKey(wait_until_next_frame) & 0xFF == ord("q"):
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
        "../data/Penalty_Neymar.mp4": 1,
        "../data/Penalty_Lampard.mp4": 0,
        "../data/Penalty_Mata.mp4": 1,
        "../data/Penalty_Olic.mp4": 1,
    }
    for video_path, label in videos.items():
        dataset = analyze_video(path=video_path, output=label, data=dataset)
    dataset_array = np.array(dataset)
    np.save("../data/training_data.npy", dataset_array)


if __name__ == "__main__":
    create_training_dataset()
