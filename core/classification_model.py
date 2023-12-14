from xgboost import XGBClassifier
import numpy as np


class XGBoostClassifier:
    def __init__(self, np_array_path, num_classes, min_samples_bootstrap=10):
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
        xgbc = XGBClassifier(
            objective="multi:softprob",  # Use 'multi:softprob' for multiclass problems
            num_class=self.num_classes,  # Number of classes in the classification problem
            eval_metric="mlogloss",  # Use 'mlogloss' for multiclass logloss
        )

        xgbc.fit(self.X_train, self.y_train)
        self.classifier = xgbc

    def evaluate_train_accuracy(self):
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
        if X_sample.ndim == 1:
            X_sample = [X_sample]
        return self.classifier.predict_proba(X_sample)
