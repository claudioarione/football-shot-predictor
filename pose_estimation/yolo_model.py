import cv2


class YOLO:
    """
    The YOLO class encapsulates a pre-trained YOLO (You Only Look Once) model for object detection.

    The model is initialized with the configuration and weights files, and it can evaluate images
    to detect objects within them using the YOLOv4-tiny architecture.

    Attributes:
        network (cv2.dnn_Net): The YOLO neural network loaded from the provided configuration and weights.
        out_layers (List[str]): Names of the output layers of the YOLO network.

    Methods:
        evaluate(image): Processes an image through the network and returns the detection results.

    Example:
        yolo_detector = YOLO()
        frame_results = yolo_detector.evaluate(frame)
    """

    def __init__(
        self,
        cfg_file_path="data/training/yolov4-tiny.cfg",
        weights_file_path="data/training/yolov4-tiny.weights",
    ):
        """
        Initializes the YOLO class with the configuration and weights of the pre-trained model.

        :param cfg_file_path: Path to the YOLOv4-tiny configuration file.
        :param weights_file_path: Path to the YOLOv4-tiny weights file.
        """
        # Crea
        self.network = cv2.dnn.readNetFromDarknet(cfg_file_path, weights_file_path)
        self.network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.out_layers = self.network.getUnconnectedOutLayersNames()

    def evaluate(self, image):
        """
        Processes an image through the YOLO network and returns the detection results.

        :param image: The image to be processed by the network.
        :return: The output from the final layers of the YOLO network after processing the image.
        """
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, swapRB=True, crop=False)
        self.network.setInput(blob)
        output_from_network = self.network.forward(self.out_layers)
        return output_from_network
