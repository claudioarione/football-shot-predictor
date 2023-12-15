from pose_estimation.pose_module import PoseDetector
import numpy as np
import cv2


def check_if_stop_video(
    ball_box, image, attacker_detector: PoseDetector, threshold_in_pixels=35
):
    """
    Determines if the video playback should stop based on the proximity of the attacker's foot to the ball.

    :param ball_box: Bounding box coordinates of the ball (x, y, width, height).
    :param image: The frame from the video to analyze.
    :param attacker_detector: An instance of PoseDetector to detect the attacker's pose.
    :param threshold_in_pixels: Distance threshold in pixels to determine proximity.
    :return: Boolean indicating whether to stop the video (True if the foot is close to the ball).
    """
    if ball_box is None:
        return False

    keypoints = attacker_detector.get_position(image)
    feet_keypoints = [keypoint for keypoint in keypoints if keypoint[0] >= 27]

    for keypoint in feet_keypoints:
        keypoint_center = keypoint[1:]
        ball_center = center_of_box(ball_box)
        if euclidean_distance(keypoint_center, ball_center) < threshold_in_pixels:
            return True

    return False


def preprocess(all_features: np.array, landmarks_list: list) -> np.array:
    """
    Processes and flattens pose landmarks to be concatenated to the features array.

    :param all_features: The array accumulating all features over time.
    :param landmarks_list: The list of current frame landmarks to process.
    :return: Updated array of features with the current frame's landmarks appended.
    """
    if len(landmarks_list) == 0:
        return all_features
    landmarks_array = np.array(landmarks_list[:][:])
    frame_features = landmarks_array[:, 1:].flatten()
    all_features = np.concatenate([all_features, frame_features])
    return all_features


def process_player(frame, player, detector: PoseDetector) -> None:
    """
    Processes a single player's image to estimate the pose using the detector.

    :param frame: the frame of the video
    :param player: the player to estimate the pose
    :param detector: the mediapipe pose detector
    """
    if not player:
        return
    x, y, w, h = player
    player_image = frame[y : y + h, x : x + w]
    detector.find_pose(player_image)


def center_of_box(box) -> np.array:
    """
    Calculates the center point of a given bounding box.

    :param box: coordinates of the area around the object
    :return: coordinates of the center of the box
    """
    x, y, width, height = box
    return np.array([x + width / 2, y + height / 2])


def euclidean_distance(center1, center2):
    """
    Computes the Euclidean distance between two points.

    :param center1: The first point (x, y).
    :param center2: The second point (x, y).
    :return: Euclidean distance between the two points.
    """
    return np.linalg.norm(center1 - center2)


def find_similar_boxes(
    current_frame_boxes, previous_attacker, previous_goalkeeper, threshold: int = 40
):
    """
    Finds boxes in the current frame similar to the previous attacker and goalkeeper positions.

    :param current_frame_boxes: Detected boxes in the current frame.
    :param previous_attacker: Previous attacker's bounding box.
    :param previous_goalkeeper: Previous goalkeeper's bounding box.
    :param threshold: Distance threshold to consider boxes as similar.
    :return: List of similar boxes found in the current frame.
    """
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
    """
    Applies non-maxima suppression to filter out overlapping boxes based on their class scores.

    :param boxes: List of detected bounding boxes.
    :param class_scores: List of scores corresponding to the detected boxes.
    :return: List of filtered bounding boxes after applying non-maxima suppression.
    """
    chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.4)
    boxes = [boxes[i] for i in chosen_boxes]
    return boxes


def draw_text_with_background(
    image,
    text,
    position,
    font,
    scale,
    color,
    thickness=1,
    bg_color=(255, 255, 255),
    text_padding=5,
):
    """
    Draws text on an image with a background rectangle for better visibility.

    :param image: The image to draw on.
    :param text: The text to be drawn.
    :param position: The bottom-left corner position of the text.
    :param font: Font type for the text.
    :param scale: Font scale for the text size.
    :param color: Color of the text.
    :param thickness: Thickness of the text stroke.
    :param bg_color: Background color for the text.
    :param text_padding: Padding around the text for the background rectangle.
    """
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    text_width, text_height = text_size
    lower_left = (position[0] - text_padding, position[1] + text_padding)
    upper_right = (
        position[0] + text_width + text_padding,
        position[1] - text_height - text_padding,
    )
    cv2.rectangle(image, lower_left, upper_right, bg_color, -1)
    cv2.putText(image, text, position, font, scale, color, thickness)


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
    thickness = 2
    image_with_box = cv2.rectangle(
        image_to_process, start_point, end_point, color, thickness
    )

    return image_with_box


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
