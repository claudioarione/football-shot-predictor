# Football Shot Predictor

## Introduction
This project leverages advanced computer vision techniques to analyze and predict the direction of football penalty kicks. Utilizing a combination of YOLO for objects detection and MediaPipe for pose estimation, our system processes video footage to forecast the outcomes of shots and goalkeeper dives.

## Features
- Advanced objects detection using YOLO
- Pose estimation with MediaPipe
- Predictive modeling with XGBoost for shot direction
- Visual results with intuitive shot and dive predictions

## Requirements
- Python 3.*
- Pre-trained YOLO and MediaPipe models
- XGBoost for classification

## Installation
1. Clone the repository.
2. Install dependencies: `pip3 install -r requirements.txt`.

## Usage
Run the main application with the following command:

```shell
python3 main.py --arguments*
```
### Arguments explained
- Use the following flag without additional arguments to train the model using the default video list:
`-t or --train`

- Use the following flag without additional arguments to predict outcomes using the default video list:
`-p or --predict`

- Use the following argument with the path to a file containing your custom list of videos for training or prediction:
`-f or --filepath`

- Provide the path to the folder containing numpy arrays from previous training if you want to use them for prediction:
`--training_data_load`

- Specify the folder in which to save new training data arrays:
`--training_data_save`

## Detailed Methodology
For an in-depth explanation of our methodology, including player detection, pose estimation, data preparation, and feature extraction, please refer to our comprehensive report.

## Results
Our system demonstrates the potential of computer vision in sports analytics, offering novel insights into the dynamics of football penalties.

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Contributors
- [Claudio Arione](https://github.com/claudioarione)
- [Riccardo Inghilleri](https://github.com/riccardoinghilleri)

## Acknowledgements
- Prof. Mathieu Bredif for guidance and support throughout the project.

---

For more details on our project, including possible improvements and future work, please see our full [report](https://github.com/claudioarione/football-shot-predictor/blob/master/documentation/report.pdf).

