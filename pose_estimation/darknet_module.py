import cv2
import numpy as np


def draw_object_bounding_box(image_to_process, box):
    """
    Drawing object borders with captions
    :param image_to_process: original image
    :param box: coordinates of the area around the object
    :return: image with marked objects
    """
    x, y, w, h = box
    start = (x, y)
    end = (x + w, y + h)
    color = (0, 255, 0)
    width = 2
    final_image = cv2.rectangle(image_to_process, start, end, color, width)

    return final_image


def identify_players(boxes):
    # For simplicity, only consider the two largest detected persons, assuming they are the goalkeeper and the attacker
    sorted_boxes = sorted(boxes, key=lambda x: x[2] * x[3], reverse=True)
    # The player closer to the center of the goal (typically upper half of the image) is likely the goalkeeper
    # FIXME until now, the attacker is the biggest, the keeper is the 2nd biggest. There are some
    #  problems after the player has shot the ball
    goalkeeper = sorted_boxes[1]
    attacker = sorted_boxes[0]

    return goalkeeper, attacker


def non_maxima_suppression(boxes, class_scores):
    chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.4)
    boxes = [boxes[i] for i in chosen_boxes]
    return boxes


def identify_all_persons(outputs, shape, threshold=0):
    """
    :param outputs: 2D NumPy array derived from a CNN
    :param shape: a tuple having the width as first coordinate and the height as second
    :param threshold: the minimum value of the classification score to be accepted
    :return: a tuple consisting of an array containing the coordinates x, y, width and height of all
             the identified human beings and an array of the respective confidence score
    """
    width, height = shape
    class_scores, boxes = [], []

    for out in outputs:
        for obj in out:
            scores = obj[5:]
            class_index = np.argmax(scores)
            class_score = scores[class_index]

            if class_index != 0:
                # The identified object is not a person
                continue

            if class_score > threshold:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                obj_width = int(obj[2] * width)
                obj_height = int(obj[3] * height)

                box = [center_x - obj_width // 2, center_y - obj_height // 2,
                       obj_width, obj_height]
                boxes.append(box)
                class_scores.append(float(class_score))

    return boxes, class_scores


class YOLO:

    def __init__(self):
        # Create net from the given config file and weights, belonging to a pre-trained
        # YOLO model
        self.network = cv2.dnn.readNetFromDarknet('../data/training/yolov4-tiny.cfg',
                                                  '../data/training/yolov4-tiny.weights')
        # Set CPU as the target - TODO use GPU
        self.network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        # Set up the names of the final layers of the network
        self.out_layers = self.network.getUnconnectedOutLayersNames()

    def evaluate(self, image):
        # Scale factor: 1/255, i.e., scale the pixel values to [0..1]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, swapRB=True, crop=False)
        # Set input and get results
        self.network.setInput(blob)
        output_from_network = self.network.forward(self.out_layers)
        return output_from_network


def analyze_video():
    model = YOLO()

    video = cv2.VideoCapture('../data/Penalty_Neymar.mp4')
    size = (480, 288)

    # Defining loop for catching frames
    while True:
        ret, frame = video.read()
        if not ret:
            break
        # Resize the frame
        frame = cv2.resize(frame, size)

        outputs = model.evaluate(frame)
        boxes, class_scores = identify_all_persons(outputs, size)

        # Selection
        filtered_boxes = non_maxima_suppression(boxes, class_scores)
        # Identify attacker and goalkeeper
        a, g = identify_players(filtered_boxes)
        # Draw boundaries for attacker and goalkeeper
        draw_object_bounding_box(frame, a)
        draw_object_bounding_box(frame, g)

        cv2.imshow('Image', frame)

        cv2.waitKey(10)

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    analyze_video()
