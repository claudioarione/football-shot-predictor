from xgboost import XGBClassifier
import numpy as np


class XGBoostClassifier:
    """
    A classifier based on the XGBoost machine learning framework, designed for training and
    predicting with XGBoost models.

    Attributes:
        X_train (np.array): Feature set for training the model.
        y_train (np.array): Labels corresponding to the feature set.
        num_classes (int): Number of classes to predict.
        classifier (XGBClassifier): Trained XGBClassifier instance.

    Methods:
        train_model(): Trains the XGBoost classifier with the provided training data.
        evaluate_train_accuracy(): Evaluates the accuracy of the trained model on the training data.
        predict_class(X_sample): Predicts class probabilities for given samples.
    """

    def __init__(self, np_array_path, num_classes, min_samples_bootstrap=10):
        """
        Initializes the XGBoost classifier with training data loaded from a NumPy array file and
        bootstraps additional samples if necessary.

        :param np_array_path: Path to the NumPy array file containing the training data.
        :param num_classes: Number of classes in the classification problem.
        :param min_samples_bootstrap: Minimum number of samples required for bootstrapping.
        """
        data = np.load(np_array_path)
        self.X_train = data[:, :-1]
        self.y_train = data[:, -1]
        self.num_classes = num_classes
        if self.X_train.shape[0] != self.y_train.shape[0]:
            raise Exception("X_train and y_train must have the same number of samples")
        if self.X_train.shape[0] < min_samples_bootstrap:
            for i in range(min_samples_bootstrap - self.X_train.shape[0]):
                index = np.random.randint(0, self.X_train.shape[0])
                line_to_bootstrap = self.X_train[index]
                X_bootstrapped = line_to_bootstrap + np.random.normal(
                    0, 0.1, line_to_bootstrap.shape[0]
                )
                X_bootstrapped = np.array([int(round(el)) for el in X_bootstrapped])
                self.X_train = np.vstack((self.X_train, X_bootstrapped))
                self.y_train = np.append(self.y_train, self.y_train[index])

    def train_model(self):
        """
        Trains the XGBoost classifier using the X_train and y_train attributes.
        """
        xgbc = XGBClassifier(
            objective="multi:softprob",  # Use 'multi:softprob' for multiclass problems
            num_class=self.num_classes,  # Number of classes in the classification problem
            eval_metric="mlogloss",  # Use 'mlogloss' for multiclass logloss
        )

        xgbc.fit(self.X_train, self.y_train)
        self.classifier = xgbc

    def evaluate_train_accuracy(self):
        """
        Evaluates the training accuracy of the classifier.

        :return: Accuracy score as a float.
        """
        if not hasattr(self, "classifier"):
            print("Cannot evaluate accuracy before training the model")
            return 0

        predictions_matrix = self.classifier.predict(self.X_train)
        predictions_vector = [
            [index for index, value in enumerate(list_m) if value == 1]
            for list_m in predictions_matrix
        ]
        predictions_vector = [
            el for list_pred in predictions_vector for el in list_pred
        ]

        return np.mean(self.y_train == predictions_vector)

    def predict_class(self, X_sample):
        """
        Predicts class probabilities for the given samples using the trained XGBoost classifier.

        :param X_sample: A single sample or an array of samples for which to predict probabilities.
        :return: An array of predicted class probabilities.
        """
        if X_sample.ndim == 1:
            X_sample = [X_sample]
        return self.classifier.predict_proba(X_sample)
