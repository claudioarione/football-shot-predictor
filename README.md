# Football Shot Predictor
[![License: MIT][license-image]][license]
## Introduction
This project leverages advanced computer vision techniques to analyze and predict the direction of football penalty kicks. Utilizing a combination of YOLO for objects detection and MediaPipe (Google) for pose estimation, our system processes video footage to forecast the outcomes of shots and goalkeeper dives.

https://github.com/claudioarione/football-shot-predictor/assets/100593859/95c21445-a00e-4f41-8ec2-e7f0b0d52561

## The Team
- [Claudio Arione](https://github.com/claudioarione)
- [Riccardo Inghilleri](https://github.com/riccardoinghilleri)

## Features

| Functionality                    | Description                                                                                                                                                                 |
|:---------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Objects detection using YOLO     | You only look once (YOLO) is a state-of-the-art, real-time object detection system. We used it to detect the boundin boxes for the attacker, goalkeeper and the soccer ball |
| Pose estimation with Mediapipe   | Mediapipe by Google is the most accurate and efficient technology to perform pose estimation (33 keypoints).                                                                |
| Predictive modeling with XGBoost | Strategic extraction of pose landmark features precedes the kick, forming the backbone of our dataset                                                                       |

## Usage

### Requirements
1. `Python 3.*` 
   
   You can check if you have already installed python by opening the terminal \faicon{terminal} and using the following command:
   ```shell
   python --version
   ```
2. Clone this repository using the following command (or using ssh):
   ```shell
   git clone https://github.com/claudioarione/football-shot-predictor
   ```
3. Install the needed libraries going to the folder containing the requirements.txt file:
   ```shell
   pip3 install -r requirements.txt
   ```

### Running
To run the code and utilize the football shot predictor, follow these steps:
#### Running the code
1. Open your terminal or command prompt
2. Navigate to the directory where the **main.py** file is located
3. To run the program, type:
   ```shell
   python3 main.py --arguments*
   ```
   followed by the specific **arguments** you want to use.
#### Arguments explained
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

## Results
Our system demonstrates the potential of computer vision in sports analytics, offering novel insights into the dynamics of football penalties.

## Acknowledgements
- Prof. Mathieu Bredif for guidance and support throughout the project.

## Copyright and License

Football Shot Predictor is copyrighted 2023.

Licensed under the **[MIT License][license]**;
you may not use this software except in compliance with the License.

[license]: https://github.com/claudioarione/football-shot-predictor/blob/master/LICENSE
[license-image]: https://img.shields.io/badge/License-MIT-blue.svg

---

For more details on our project, including possible improvements and future work, please see our full [report](https://github.com/claudioarione/football-shot-predictor/blob/master/documentation/report.pdf).

