import sys, pandas as pd
import training, predict
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Football Shot Predictor",
        description="Predicts the expected direction of the shot and of the goalkeeper",
        epilog="Visit our GitHub repository for further information"
    )

    parser.add_argument('-t', '--train', action='store_true',
                        help='Train model on specified list of videos or on default one')
    parser.add_argument('-p', '--predict', action='store_true',
                        help='Predict shot outcome on specified list of videos or on default one')
    parser.add_argument('-f', '--filepath', action='store',
                        help='Path of the file containing the list of videos')
    parser.add_argument('--training_data_load', action='store',
                        help='Path of the .npy file containing the data obtained by a previous training that should be'
                             'used in prediction')
    parser.add_argument('--training_data_save', action='store',
                        help='Path where to save the newly created training set')

    args = parser.parse_args()

    if args.train and args.predict:
        sys.exit('Cannot pass both train and predict arguments')
    elif args.train:
        # Train model
        videos_link_file = args.filepath or 'data/training/videos_link_list.csv'
        save_training_data_path = args.training_data_save or 'data/training_data.npy'
        training_data = pd.read_csv(videos_link_file, dtype={0: str, 1: int})
        training.create_training_dataset(training_data, save_training_data_path)
    else:
        # Predict outcome of the shot
        videos_link_file = args.filepath or 'data/predict/videos_link_list.txt'
        training_data_path = args.training_data_load or 'data/training_data.npy'
        with open(videos_link_file, 'r') as videos_file:
            videos = [video.strip() for video in videos_file.readlines()]
        predict.show_predictions(videos, training_data_path)
