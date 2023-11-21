import cv2
import mediapipe as mp
import time


class PoseDetector:
    """
    A class to detect human poses in images or videos using MediaPipe.
    """

    # TODO: analyze better the params of Pose class
    def __init__(
        self,
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        """
        Initializes the pose detector with the specified parameters.

        :param static_image_mode: Whether to treat the input images as a batch of static and possibly unrelated images,
            or a stream of images where each image is related to the previous one.
        :param model_complexity: Complexity of the pose landmark model: 0, 1 or 2.
        :param smooth_landmarks: Whether to filter landmarks across different input images to reduce jitter.
        :param enable_segmentation: Whether to predict segmentation masks.
        :param smooth_segmentation: Whether to filter segmentation masks across different input images to reduce jitter.
        :param min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for the detection to be considered successful.
        :param min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the landmark tracking to be considered successful.
        """
        self.results = None
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode,
            model_complexity,
            smooth_landmarks,
            enable_segmentation,
            smooth_segmentation,
            min_detection_confidence,
            min_tracking_confidence,
        )

    def find_pose(self, img, draw: bool = True):
        """
        Processes an image and detects the human pose.

        :param img: The image on which detection is to be performed.
        :param draw: Whether to draw the landmarks and connections on the image.
        :return: The original image with landmarks and connections drawn if draw is True.
        """
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_RGB.flags.writeable = False  # only to improve performance
        self.results = self.pose.process(img_RGB)
        img_RGB.flags.writeable = True
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(
                    img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                )
        return img

    def get_position(self, image_box):
        """
        Obtains the position of human pose landmarks.

        :param image_box: The image box from which landmarks are to be identified, as an array containing,
                            in order, x, y, width and height.
        :return: A list, in which each element contains the landmark id and x and y coordinates.
        """
        landmarks_list = []
        if self.results.pose_landmarks:
            for position, landmark in enumerate(self.results.pose_landmarks.landmark):
                x, y, w, h = image_box
                cx, cy = int(x + landmark.x * w), int(y + landmark.y * h)
                landmarks_list.append([position, cx, cy])
        return landmarks_list


def test():
    # global variables
    # pose_estimator = []
    # pose_estimator_dim = []
    # # For each object detected
    # # WHICH POSE ESTIMATOR TO USE.
    # selected_pose_idx = 0

    cap = cv2.VideoCapture("../data/Penalty_Neymar.mp4")
    previous_time = 0
    detector = PoseDetector()
    while True:  # TODO: maybe is better cap.isOpened() and cap.release()
        success, img = cap.read()
        if not success:
            break
        img = cv2.resize(img, (0, 0), None, 0.7, 0.7)
        img = detector.find_pose(img)
        landmarks_list = detector.get_position(img)
        print(landmarks_list)  # debug
        # landmarks_list = detector.get_position(img, draw=False)
        # cv2.circle(img, (landmarks_list[32][1], landmarks_list[32][2]), 20, (0, 0, 255), cv2.FILLED)
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(
            img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3
        )
        cv2.imshow("Image", img)
        cv2.waitKey(int(fps) + 1)


if __name__ == "__main__":
    test()
