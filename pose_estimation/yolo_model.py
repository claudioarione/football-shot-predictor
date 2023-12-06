import cv2


class YOLO:

    def __init__(self, cfg_file_path="data/training/yolov4-tiny.cfg",
                 weights_file_path="data/training/yolov4-tiny.weights"):
        # Create net from the given config file and weights, belonging to a pre-trained
        # YOLO model
        self.network = cv2.dnn.readNetFromDarknet(
            cfg_file_path, weights_file_path
        )
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
